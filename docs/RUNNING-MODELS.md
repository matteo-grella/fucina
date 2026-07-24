# RUNNING-MODELS — CLI cheat sheet for every model in the repo

Copy-paste commands for running each supported model family from the terminal.
Everything goes through `zig build <step> -- <args>`. The commands below assume the
weights sit under `models/` — **the repo does not ship any weights**; see the next
section for where to download each artifact.

**Always build with `-Doptimize=ReleaseFast` when you care about speed** (Debug is
10-50x slower), and **build on the machine you will run on**: without `-Dtarget`
the binary is tuned to the compiling host's exact CPU; cross-compiling needs
`-Dcpu=...` too or you get baseline features and lose the fast kernels (see
`AGENTS.md`, build options). The first invocation compiles (~1 min); after that the binary is cached.
You can also call the installed binaries directly from `zig-out/bin/` (e.g.
`./zig-out/bin/fucina-qwen3 …`) to skip the build step entirely.

## Getting the weights

All LLM/ASR/TTS weights are GGUF files from Hugging Face. The `hf` CLI (from
`pip install -U huggingface_hub`; formerly `huggingface-cli`) downloads single files:

```sh
mkdir -p models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models
hf download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf --local-dir models
hf download unsloth/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-UD-Q6_K.gguf --local-dir models
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q6_K.gguf --local-dir models
```

| Family | Runner | Where to download |
| --- | --- | --- |
| Qwen3 dense (0.6B/1.7B/…) | `zig build qwen3` | Official [`Qwen/Qwen3-0.6B-GGUF`](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) / [`Qwen/Qwen3-1.7B-GGUF`](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF) ship Q8_0 only; the full K-quant ladder (Q4_K_S … Q6_K + bf16) is on [`bartowski/Qwen_Qwen3-0.6B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF) or [`unsloth/Qwen3-0.6B-GGUF`](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF) (same pattern per size). |
| Qwen3 MoE 30B-A3B | `zig build qwen3` | [`unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) — `…-Q5_K_M.gguf` (21.7 GB), `…-Q6_K.gguf` (25.1 GB). |
| Qwen3.5 0.8B (Gated-DeltaNet hybrid) | `zig build qwen35` | [`unsloth/Qwen3.5-0.8B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) — `Qwen3.5-0.8B-Q8_0.gguf` (812 MB); also mirrored by lmstudio-community and bartowski. |
| Gemma 4 26B-A4B (MoE) | `zig build gemma4` | [`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) — `gemma-4-26B-A4B-it-UD-Q6_K.gguf` (23.2 GB). |
| DiffusionGemma 26B-A4B | `zig build diffusion-gemma` | [`unsloth/diffusiongemma-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) — `diffusiongemma-26B-A4B-it-Q6_K.gguf` (22.7 GB), `…-Q4_K_M.gguf` (16.8 GB). |
| OmniVoice TTS | `zig build omnivoice` | [`Serveurperso/OmniVoice-GGUF`](https://huggingface.co/Serveurperso/OmniVoice-GGUF) — `omnivoice-base-*.gguf` + `omnivoice-tokenizer-*.gguf` (F32/BF16/Q8_0/Q4_K_M each). |
| Parakeet ASR (NeMo FastConformer) | `zig build parakeet` | [`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf) — e.g. `tdt_ctc-110m-f16.gguf` (267 MB), `tdt-0.6b-v3-f16.gguf` (1.44 GB); f16/q8_0/q6_k/q5_k/q4_k per model. |
| NAM amp profiles | `zig build nam` | [TONE3000 (formerly ToneHunt)](https://tonehunt.org/) — free community `.nam` captures; any upstream-format 0.5.0–0.7.x profile loads. |

Notes:

- **Gemma license.** Gemma-family weights (Gemma 4, DiffusionGemma) are distributed
  under Google's Gemma Terms of Use. The `google/…` originals on Hugging Face are gated
  behind accepting those terms; the unsloth GGUF conversions were not gated at the time
  of writing, but the terms still apply to the weights either way.
- **Filenames.** The commands below use the upstream filenames for the artifacts above.
  bartowski prefixes files with `Qwen_` (e.g. `Qwen_Qwen3-0.6B-Q4_K_S.gguf`) — every
  runner takes a plain path, so rename or adjust as you like.
- **f16 Qwen3.** Some commands reference `Qwen3-0.6B-f16.gguf`; if your source only
  ships bf16, transcode one locally:
  `zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf <src>.gguf --out models/Qwen3-0.6B-f16.gguf --dtype f16`.

---

## The examples

Every build step below has its own getting-started README next to its entry
file — `examples/<folder>/README.md` — and that is where the copy-paste
commands, weight pointers, per-runner flags, and parity/bench harnesses now
live. The table maps each step to its README. To get a first feel for the
engine, run the smallest chat model as a REPL:

```sh
# Multi-turn interactive REPL (empty line or Ctrl-D quits); the full qwen3
# command set is in examples/qwen3/README.md
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --repl
```

| Build step | What it is | Getting started |
| --- | --- | --- |
| `zig build qwen3` | Qwen3 dense (0.6B/1.7B/…) + MoE 30B-A3B: chat/REPL, speculative decoding, benchmarks, parity tooling | [examples/qwen3/README.md](../examples/qwen3/README.md) |
| `zig build deepseek2` | DeepSeek V2/V3 MLA family (V2-Lite, Moonlight-16B-A3B) | [examples/deepseek2/README.md](../examples/deepseek2/README.md) |
| `zig build glm4moe` | GLM-4.5 family with native `--mtp` multi-token-prediction drafting | [examples/glm4moe/README.md](../examples/glm4moe/README.md) |
| `zig build deepseek4` | DeepSeek V4 Flash 284B-A13B (streamed experts, MTP sidecar) | [examples/deepseek4/README.md](../examples/deepseek4/README.md) |
| `zig build gemma4` | Gemma 4 26B-A4B MoE chat/REPL | [examples/gemma4/README.md](../examples/gemma4/README.md) |
| `zig build diffusion-gemma` | DiffusionGemma 26B-A4B block text-diffusion (live inline denoising) | [examples/diffusion_gemma/README.md](../examples/diffusion_gemma/README.md) |
| `zig build qwen35` | Qwen3.5 0.8B Gated-DeltaNet hybrid + Ternary-Bonsai-27B loader/parity harness | [examples/qwen35/README.md](../examples/qwen35/README.md) |
| `zig build inkling` | Inkling 975B-A41B hybrid attention + MoE, text + image/audio towers | [examples/inkling/README.md](../examples/inkling/README.md) |
| `zig build lmserve` | OpenAI-compatible HTTP server over the family backends | [examples/lmserve/README.md](../examples/lmserve/README.md) |
| `zig build omnivoice` | OmniVoice MaskGIT TTS: voice cloning/design, codec round-trip | [examples/omnivoice/README.md](../examples/omnivoice/README.md) |
| `zig build parakeet` | Parakeet ASR (NeMo FastConformer): transcribe/stream/mic | [examples/parakeet/README.md](../examples/parakeet/README.md) |
| `zig build locate-anything` | LocateAnything-3B open-vocabulary detection | [examples/locate_anything/README.md](../examples/locate_anything/README.md) |
| `zig build facedetect` | Face detection/recognition (buffalo_l): detect/embed/verify/analyze | [examples/facedetect/README.md](../examples/facedetect/README.md) |
| `zig build nam` | Neural Amp Modeler: `.nam` profiles, live amp sim, training | [examples/nam/README.md](../examples/nam/README.md) |
| `zig build nanochat` | nanochat port: tok-train/base-train/sft/eval-bpb/chat | [examples/nanochat/README.md](../examples/nanochat/README.md) |
| `zig build finetune` | LoRA fine-tune a Qwen3 GGUF on CPU; merge/serve via `export-gguf` | [examples/finetune/README.md](../examples/finetune/README.md) |
| `zig build es-finetune` | Gradient-free (evolution strategies) fine-tune, LoRA or full-parameter | [examples/es_finetune/README.md](../examples/es_finetune/README.md) |
| `zig build cartridge` | Train a corpus into a reusable KV prefix (self-study distillation) | [examples/cartridge/README.md](../examples/cartridge/README.md) |
| `zig build cartridge-fleet` | One cartridge per document + cosine cartridge-RAG serving | [examples/cartridge_fleet/README.md](../examples/cartridge_fleet/README.md) |
| `zig build engram` | Conditional n-gram memory grafted onto a frozen Qwen3 GGUF | [examples/engram/README.md](../examples/engram/README.md) |
| `zig build spirals` | Two-spirals MLP training demo (SGD/AdamW/Muon/APOLLO); no downloads | [examples/spirals/README.md](../examples/spirals/README.md) |
| `zig build es-spirals` | Two-spirals from scratch with evolution strategies | [examples/es_spirals/README.md](../examples/es_spirals/README.md) |
| `zig build es-ternary-spirals` | Ternary-native ES: packed TQ2_0 genome = the inference model | [examples/es_ternary_spirals/README.md](../examples/es_ternary_spirals/README.md) |
| `zig build ptqtp-spirals` | Float MLP post-training-quantized to dual trit-planes | [examples/ptqtp_spirals/README.md](../examples/ptqtp_spirals/README.md) |
| `zig build ptqtp-qwen3` | PTQTP-decorate a Qwen3 GGUF's linears; NLL before/after | [examples/ptqtp_qwen3/README.md](../examples/ptqtp_qwen3/README.md) |
| `zig build smoke` | smoke: the minimal tensor/autograd sanity demo | [examples/smoke/README.md](../examples/smoke/README.md) |

---

## Shared machinery

Cross-runner machinery documented once. The per-example READMEs link back
here instead of repeating it.

### Streaming MoE experts from disk (out-of-core: models bigger than RAM)

`--moe-stream` keeps only the dense weights resident and reads the routed
experts on demand from the GGUF through a tiered store — pinned hot set →
per-layer LRU cache → `pread` — so a mixture model whose expert stacks dwarf
physical RAM still loads and decodes. Output is bit-identical to the resident
path (same blocks, same kernels); the price is disk reads on cache misses,
i.e. tokens per second. llama.cpp split GGUFs (`-00001-of-0000N`) load
transparently.

```sh
# Qwen3-235B-A22B Q4_K_M (142 GB, 3 split parts) on a 64 GB machine:
# point at part 1; ~24 GB peak RSS with a 20 GB expert budget.
zig build qwen3 -Doptimize=ReleaseFast -- \
  models/Qwen3-235B-A22B-Instruct-2507-Q4_K_M-00001-of-00003.gguf \
  --prompt "The three most important ideas in computer science are" \
  --gen 64 --moe-stream --moe-cache-mb=20480

# Same, with router-lookahead readahead and warm-restart chat
zig build qwen3 -Doptimize=ReleaseFast -- <part1.gguf> \
  --chat "..." --moe-stream --moe-cache-mb=20480 --moe-pilot --kv-save
```

Measured on that 235B/64 GB configuration (M1 Max): cold 0.66 tok/s at a
52% hit rate (131 GB streamed for prefill + 24 tokens); the second run
auto-pins the hottest 944 experts (10.7 GB) from the recorded usage history
and reaches 0.81 tok/s at 59% — the engine gets faster the more it is used.
On the 30B MoE (fits in RAM), streaming trades nothing but speed: byte-
identical greedy output at half the resident-path RSS.

Knobs:

- `--moe-cache-mb=N` / `--moe-cache-slots=N` — RAM budget for the streamed
  tiers (default: half of available memory) or a fixed per-layer LRU size.
- `--moe-pin-mb=N` / `--moe-no-learn` — pinned-tier budget (default: half
  the budget once the sidecar `<gguf>.experts` histogram has enough
  history), or disable the learning cache entirely.
- `--moe-pilot` — router lookahead: predict the next layer's experts from
  the current post-attention state and readahead them from a background
  I/O thread while the current layer computes. Measured recall of the
  one-layer-ahead prediction: 87.6% (30B), 90.5% (235B). Never changes
  output.
- `--moe-expert-top-p=F` — routing sparsification: keep experts per token
  up to cumulative router weight F and skip the rest (30B: F=0.7 cut disk
  traffic 55%). Quality-traded; `F>=1` is the exact baseline.
- `--moe-mirror=PATH` — another full copy of the model, typically on
  another drive (repeatable: one flag per copy; for split GGUFs point at
  the copy's part 1). Expert reads split across every copy by a
  deterministic per-expert hash, so miss-bound streaming aggregates each
  drive's bandwidth; output is unchanged, and a mirror read error falls
  back to the primary. `--moe-mirror-weights=W1,W2,...` biases the split
  for asymmetric drives (share relative to the primary's 1; default an
  even split). The exit stats report the per-copy split.
- `--moe-io-threads=N` — demand-miss reads fan out across N persistent
  I/O worker threads (default 8; the forward thread participates too;
  0 = sequential). Parallel misses are what turn disk queue depth — and
  mirror copies — into real aggregate bandwidth within one token's
  expert fetches. Output is unchanged.
- `--kv-save[=PATH]` — crash-safe KV persistence for `--chat`/`--repl`:
  conversations reopen warm across process restarts with zero re-prefill
  (essential below 1 tok/s). Default sidecar `<gguf>.kvcache`.

The DeepSeek-family runners share this streamed-expert machinery; all of
them accept `--moe-stream`/`--moe-cache-mb`
([examples/deepseek2/README.md](../examples/deepseek2/README.md),
[examples/glm4moe/README.md](../examples/glm4moe/README.md),
[examples/deepseek4/README.md](../examples/deepseek4/README.md)). DeepSeek
V4 Flash's 164.6 GB Q4K release decodes on a 64 GB machine with
`--moe-stream` (measured 1.5–3.6 tok/s warm at a 20 GB expert budget).

**Faster streaming via tie-fitted ternary experts** (qwen3 families): a
streamed model gets both smaller and faster if the expert mass is
PTQTP-quantized with tied scales first, then streamed as above:

```sh
# 1. Quantize the routed experts to tied K=2 planes (~4.1 bpw), one
#    tensor at a time — source size does not need to fit in RAM.
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf model-BF16.gguf --out model-ptqtp2-tied.gguf \
  --ptqtp=2 --ptqtp-tie --ptqtp-include ffn_gate_exps,ffn_up_exps,ffn_down_exps

# 2. Stream the output exactly like any other GGUF.
zig build qwen3 -Doptimize=ReleaseFast -- model-ptqtp2-tied.gguf \
  --prompt "..." --gen 64 --moe-stream --moe-cache-mb=6144
```

Tied K=2 halves the bytes read per expert miss vs a q8_0 source, and
every cache-hit expert dot runs the folded one-pass ternary kernel
(2.09-2.37× the two-pass dot; the store folds the planes into the cache
slab at fill). `--ptqtp=3 --ptqtp-tie` is the near-parity quality shape
instead (two-pass serving). The full walkthrough with measured numbers
is [`PTQTP-RECIPE.md`](PTQTP-RECIPE.md).

### Native MTP drafting

Models that ship their own multi-token-prediction head draft with it and
verify with the trunk — lossless speculative decoding with no external
drafter:

- GLM-4.5 family: `--mtp[=depth]` drafts with the MTP head and verifies
  with one batched trunk step — lossless (byte-identical to plain greedy),
  measured 2.29 tokens per forward at depth 2 on GLM-4.5-Air Q6_K streamed
  on a 64 GB machine. The verify runs kernel-pinned (batched quant
  matmuls reproduce single-token numerics bitwise), so depth now caps at
  8 instead of the old drift-bound 2; bare `--mtp` stays depth 2.
  ([examples/glm4moe/README.md](../examples/glm4moe/README.md))
- DeepSeek V4 Flash: `--mtp=<sidecar.gguf>` — the 3.8 GB sidecar GGUF
  drafts, the trunk verifies in one batched step — lossless, measured
  84.6% draft acceptance / 1.60 tokens per trunk forward at depth 1.
  ([examples/deepseek4/README.md](../examples/deepseek4/README.md))

### Constrained decoding (`-Dllguidance=true`)

qwen3, gemma4 and lmserve share the llguidance-backed grammar engine:
`--json-schema J|@F` / `--lark G|@F` / `--regex P` (mutually exclusive;
`@F` reads the grammar from a file) on the CLI runners, `response_format`
over HTTP (design record: `docs/CONSTRAINED-DECODING.md`). The commands
below are qwen3's; gemma4 mirrors them on `--chat`/`--repl`, and lmserve
serves the same grammars for every backend it fronts, including
Ternary-Bonsai-27B ([examples/qwen35/README.md](../examples/qwen35/README.md)).

```sh
# Constrained decoding (needs a -Dllguidance=true build — REFERENCE.md §13.6):
# the reply must satisfy a JSON schema, regex, or Lark grammar. Combine with
# --no-think (the grammar governs the whole reply, thinking channel included);
# works in --chat/--repl/--prompt, composes with --spec (grammar-forced spans
# draft themselves — REFERENCE.md 13.9.6) and with --streams (one grammar
# clone per stream). Prefer sampling over --temp 0 and bound open-ended
# fields (maximum/maxLength/{m,n}) — a greedy argmax inside an unbounded
# field can loop until the token budget.
zig build qwen3 -Dllguidance=true -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "Give me facts about Paris." --no-think \
  --json-schema '{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer","maximum":99999999}},"required":["city","population"],"additionalProperties":false}'
zig build qwen3 -Dllguidance=true -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "What is the capital of Italy? One word." --no-think --temp 0 --regex '[A-Z][a-z]{2,15}'
```

### `--experts=borrow` (zero-copy MoE expert load)

Maps the MoE experts zero-copy instead of x4-packing them. Q6_K experts
otherwise copy+widen ~20 GB on load (slow, doubles memory, can swap on
<48 GB boxes); borrow loads in ~2-3 s at ~half the RSS. Default is pack
(peak CPU throughput). Numerically identical (same parity-tested kernels).
Accepted by gemma4, diffusion-gemma and lmserve; gemma MoE fleets served
by `lmserve --fleet` need it, and cartridge-fleet borrows expert blocks
zero-copy on its own.

---

## GPU offload (`-Dgpu=metal` on macOS, `-Dgpu=cuda` on Linux/NVIDIA)

Two providers share one contract (gates + CPU fallback; `FUCINA_GPU=0` kill
switch, `FUCINA_GPU_TRACE=1` dispatch tracing). CUDA extras:

```sh
# Linux/NVIDIA: no CUDA SDK needed at build time (dlopen'd cuBLAS + vendored
# PTX kernels). Dense quantized prefill offloads automatically; measured on a
# RTX 5000 Ada laptop + i9-13950HX, Qwen3-4B Q4_K_M, 841-token prompt:
# prefill 109 -> 190 tok/s vs -Dgpu=none.
zig build qwen3 -Dgpu=cuda -Doptimize=ReleaseFast -- models/Qwen3-4B-Q4_K_M.gguf --chat "..."

# CUDA also keeps Q5_K weights resident and offloads their dense prefill.
# On Qwen3-0.6B-Q5_K_M, warm pp32 measured 503 -> 770 tok/s and pp128
# 620 -> 1167 tok/s on the same host.
zig build qwen3 -Dgpu=cuda -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q5_K_M.gguf --chat "..."

# Experimental decode offload (m<=8 over resident weights; Q5_K uses GEMV
# for m<4 and tiled MMA for m=4..8):
# measured 12.7 -> 33.4 tok/s decode on the same rig. Opt-in while the
# CPU/GPU quantized paths remain tolerance-equivalent rather than bit-identical.
FUCINA_GPU_DECODE=1 zig build qwen3 -Dgpu=cuda -Doptimize=ReleaseFast -- models/Qwen3-4B-Q4_K_M.gguf --chat "..."

# CUDA-only env knobs: FUCINA_GPU_TF32=1 (TF32 tensor cores for f32 GEMM,
# 2.5-3x, changes numerics), FUCINA_GPU_MIN_WORK_TRANSIENT (work floor for
# non-resident f32 operands), FUCINA_GPU_VRAM_BUDGET (bytes, tracked),
# FUCINA_GPU_QUANT_MMA=0 (diagnostic scalar fallback for quantized prefill),
# FUCINA_GPU_QUANT_SPLIT_K=0 (diagnostic unsplit tensor-core prefill),
# FUCINA_GPU_MIN_WORK_DENSE_Q5 (Q5_K prefill gate, default 2^24),
# FUCINA_GPU_MIN_WORK_DECODE_Q5 (Q5_K decode gate, default 3*2^23),
# FUCINA_GPU_KERNELS=src (NVRTC-recompile the vendored kernels).
```

### Metal specifics (`-Dgpu=metal`) — what it does and doesn't speed up

```sh
# Dense f16 models: prefill GEMMs offload automatically — Qwen3-0.6B-f16 pp512
# measured 488 -> 993 tok/s (2.03x). Decode (m=1) correctly stays on CPU.
zig build qwen3 -Dgpu=metal -Doptimize=ReleaseFast -- models/Qwen3-0.6B-f16.gguf --chat "..."

# Gemma 4 / DiffusionGemma MoE (Q6_K + Q8_0 experts): the batched expert FFN runs as
# grouped dequant-in-kernel Metal GEMMs straight off the quantized blocks — no flag
# needed. On gpu builds the loader also keeps ONE raw expert representation instead
# of the x4 packs: load drops ~40 s -> seconds, ~24 GB less memory, and CPU decode
# gets faster too (fewer weight bytes). Measured (M1 Max, macOS 14.6): MoE prefill
# section ~2.2 s -> ~1.2 s/pass, diffusion ~4.6 s/denoise-step vs 5.8 CPU.
# macOS 14 re-wires GPU buffers under CPU memory traffic (~0.5 s/pass of the
# remainder); macOS 15 residency sets should roughly halve the MoE section again.
zig build gemma4 -Dgpu=metal -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf --chat "..."
# FUCINA_GPU_MIN_WORK_QMOE=<n> tunes the MoE offload gate (default 2^30 m·n·k/layer);
# FUCINA_GPU_QMOE_MIN_FILL=<pct> = min 32-row-tile occupancy for GPU MoE dispatch (default 50;
#   0 restores pre-gate behavior; >100 forces CPU) — mid-batch expert GEMMs run ~1.8-2.4x faster
#   on CPU when tiles would be mostly empty (measured 2026-07-03, M1 Max);
# FUCINA_GPU_DEBUG=1 logs per-dispatch wall/gpu/sched times.

# Dense quantized linears (qwen3/gemma Q4_K/Q6_K/Q8_0 prefill projections) DO offload via the
# dequant-in-kernel gemmQuantNt path (weights.linearSeqQ* -> ExecContext.denseQuantMatmulGpu,
# per-format FUCINA_GPU_MIN_WORK_DENSE_Q4/Q6/Q8 gates against the packed CPU fallback;
# stable RHS residency; eager submit with deferred host visibility; ~+24-33% pp on 0.6B-Q4_K).
# Q5_K, decode (m=1), and training stay on CPU. For diffusion-gemma, --gpu-f16 additionally
# dequantizes the dense weights (attn projections, shared FFN, lm head; +~4.6 GB resident):
zig build diffusion-gemma -Dgpu=metal -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "..." --gpu-f16
# Training f32 GEMMs offload automatically when big enough (same 2^30 work gate).
```

# Global knobs (any runner)

```sh
# Thread count: default 8 (M1 Max P-cores). Lower at runtime:
FUCINA_MAX_THREADS=6 zig build qwen3 -Doptimize=ReleaseFast -- ...
# Raise ABOVE 8 only at build time: -Dmax-threads=N

# GPU GEMM offload: big f32/f16 GEMMs and dense quantized linears
# (Q4_K/Q6_K/Q8_0 on Metal; those plus Q5_K on CUDA),
# and the MoE expert FFN (Q6_K/Q8_0) run on the GPU. Metal quantized decode and training stay
# on CPU; CUDA resident f16 decode can offload, while quantized decode remains opt-in.
zig build <step> -Dgpu=metal -Doptimize=ReleaseFast -- ...
FUCINA_GPU=0 ...              # runtime kill switch
FUCINA_GPU_MIN_WORK=<n>       # f32 override (Metal default 2^32; CUDA base 2^30 plus residency policy)

# BLAS provider (default accelerate on macOS): -Dblas=none|accelerate|openblas|...
# Reference backend (slow, for validation): -Dbackend=scalar
```

Benchmark etiquette: numbers count only in `-Doptimize=ReleaseFast`, on a cool chip,
best-of-N — see `BENCHMARK.md` before making perf claims.
