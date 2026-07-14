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
`./zig-out/bin/fucina-zig-qwen3 …`) to skip the build step entirely.

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

## Qwen3 (dense 0.6B/1.7B + MoE 30B-A3B) — `zig build qwen3`

The most complete runner: chat, REPL, raw generation, speculative decoding, benchmarks,
logit-parity tooling.

```sh
# Single-turn chat (streams the reply; --no-think skips the <think> phase)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "What is the capital of France?" --no-think

# Multi-turn interactive REPL (empty line or Ctrl-D quits)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --repl

# Chat with a system prompt + sampling overrides
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "Tell me a joke" --system "You are a pirate." \
  --temp 0.7 --top-k 40 --top-p 0.9 --seed 42

# The big MoE (20 GB — give it a moment to mmap)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf \
  --chat "Explain quantum entanglement in one paragraph." --no-think

# Raw completion from a text prompt (greedy unless sampling flags given)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" --gen 64

# Lossless speculative decoding (SAM + Token-Recycling cascade; prints acceptance stats)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  --prompt "..." --gen 128 --spec

# Speculative decoding with an injected reference document (the RAG seam:
# the drafter can copy spans from the injected text)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  --prompt "Summarize the doc" --gen 128 --spec --spec-ref doc.txt

# Warm prefill/decode benchmark, fair vs llama-bench (load once, best-of-R)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf \
  <prompt-token-ids> --gen 64 --bench 5

# Batched multi-stream decode: N lockstep streams (one m=N weight pass/step)
# vs N sequential runs, aggregate tok/s + token-for-token cross-check
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  <prompt-token-ids> --gen 64 --bench 3 --streams 4

# Tokenizer-parity oracle (one token id per line; no weights loaded)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --tokenize input.txt

# Logit parity vs another implementation (raw little-endian f32 dump/compare)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  151644,872,198,9707 --logits-out /tmp/f.bin
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  151644,872,198,9707 --compare-logits /tmp/ref.bin

# q8_0 KV cache (halves KV memory — capacity option; decode is NOT faster on M1)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "..." --gen 256 --cache-type q8_0

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

Other flags: `--repeat N` (re-run forward), `--profile` (per-block timings), `--info`,
`--verify-cache N` (cached-vs-full attention check), `--min-p F`, `--repeat-penalty F`,
`--stop TOKEN_ID`, `--json-schema J|@F` / `--lark G|@F` / `--regex P` (mutually
exclusive; `@F` reads the grammar from a file).

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
- `--kv-save[=PATH]` — crash-safe KV persistence for `--chat`/`--repl`:
  conversations reopen warm across process restarts with zero re-prefill
  (essential below 1 tok/s). Default sidecar `<gguf>.kvcache`.

---

## DeepSeek family (V2/V3 MLA, GLM-4.5, V4 Flash)

Three runners share the streamed-expert machinery above; all of them accept
`--moe-stream`/`--moe-cache-mb`.

### DeepSeek V2/V3 — `zig build deepseek2`

Multi-head latent attention with the compressed KV cache as the default
(576 floats per token per layer, 8.9× smaller than reconstructed heads;
`--mla=full` selects the reconstructing path, byte-identical output) and
weight absorption folding `kv_b` into the query/value sides. Covers V2-Lite
(softmax router) and V3-style checkpoints such as Moonlight-16B-A3B
(sigmoid no-aux router, q-LoRA, MLA-native GGUF layout).

```sh
zig build deepseek2 -Doptimize=ReleaseFast -- \
  models/DeepSeek-V2-Lite-Chat.Q8_0.gguf --prompt "..." --gen 64
```

### GLM-4.5 family — `zig build glm4moe`

V3-style MoE trunk plus the model's own `nextn` multi-token-prediction
layer: `--mtp[=depth]` drafts with the MTP head and verifies with one
batched trunk step — lossless (byte-identical to plain greedy), measured
2.29 tokens per forward at depth 2 on GLM-4.5-Air Q6_K streamed on a
64 GB machine. Depth caps at 2.

```sh
zig build glm4moe -Doptimize=ReleaseFast -- \
  models/glm45-air/GLM-4.5-Air-Q6_K-00001-of-00002.gguf \
  --prompt "..." --gen 64 --mtp --moe-stream --moe-cache-mb=20480
```

### DeepSeek V4 Flash 284B-A13B — `zig build deepseek4`

The DwarfStar-class trunk: hyper-connections (4 residual streams mixed by a
Sinkhorn-normalized combine), MQA over a single 512-dim FP8-simulated KV row
with per-head sink logits, streaming compressors with an FP4/Hadamard
indexer (top-512 row selection), sqrt-softplus routing with hash-routed
early layers, and a grouped low-rank output projection. The 164.6 GB Q4K
release decodes on a 64 GB machine with `--moe-stream` (measured 1.5–3.6
tok/s warm at a 20 GB expert budget). `--chat` renders the reference
template (thinking disabled); `--vectors=DIR` replays the official-API
fixtures from a checkout of the upstream `ds4` repository and compares the
greedy continuation step by step.

```sh
zig build deepseek4 -Doptimize=ReleaseFast -- \
  models/deepseek-v4/DeepSeek-V4-Flash-Q4KExperts-...gguf \
  --chat --prompt "Answer with only the number: 2048 divided by 128 is" \
  --gen 8 --moe-stream --moe-cache-mb=20480

# Native MTP speculative decoding (the 3.8 GB sidecar GGUF drafts, the
# trunk verifies in one batched step — lossless, measured 84.6% draft
# acceptance / 1.60 tokens per trunk forward at depth 1):
zig build deepseek4 -Doptimize=ReleaseFast -- <model.gguf> --chat --prompt "..." \
  --moe-stream --mtp=models/deepseek-v4/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf

# Official-vector regression (--vectors-max-prompt raises the skip bar for
# the ~3.5k-token fixtures; --prefill-chunk sizes the batched prefill):
zig build deepseek4 -Doptimize=ReleaseFast -- <model.gguf> --moe-stream \
  --vectors=path/to/ds4/tests/test-vectors/official

# Implementation-level logit oracle: replay the upstream local-golden
# fixture (top-64 ids + raw logits at a 4096-token frontier, captured from
# a known-sane run of the same GGUF) with the upstream pass thresholds:
zig build deepseek4 -Doptimize=ReleaseFast -- <model.gguf> --moe-stream \
  --golden=path/to/ds4/tests/test-vectors/local-golden.vec
```

---

## Gemma 4 26B-A4B (MoE) — `zig build gemma4`

Same chat/REPL UX as qwen3, Gemma's `<|turn>` template, SPM tokenizer from the GGUF.
Weights: `gemma-4-26B-A4B-it-UD-Q6_K.gguf` from
[`unsloth/gemma-4-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
(subject to Google's Gemma Terms of Use — see above).

```sh
# Single-turn chat (sampling defaults come from the GGUF: temp 1.0, top-k 64, top-p 0.95)
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --chat "Why is the sky blue?"

# --experts=borrow: map the MoE experts zero-copy instead of x4-packing them.
# Q6_K experts otherwise copy+widen ~20 GB on load (slow, doubles memory, can
# swap on <48 GB boxes); borrow loads in ~2-3 s at ~half the RSS. Default is
# pack (peak CPU throughput). Numerically identical (same parity-tested kernels).
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --chat "Why is the sky blue?" --experts=borrow

# Interactive REPL with a system prompt; --think enables the thought channel
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --repl --system "Answer tersely." --think

# Lossless speculative decoding in chat/REPL (same SAM cascade + CostGate as
# qwen3; greedy output verified byte-identical with and without --spec)
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --chat "Why is the sky blue?" --spec

# Greedy decoding, capped reply length, custom stop string
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --chat "List three facts about Mars" --greedy --max 256 --stop "4."

# Encode-only (prints token ids without loading the 22 GB of weights)
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --prompt "Hello world" --tok-only

# Prefill/decode benchmark + per-block profile; logit parity from raw ids
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  2,651,235 --bench 3 --profile
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  2,651,235 --logits-out /tmp/g4.bin
```

Sampling flags mirror qwen3 (`--temp --top-k --top-p --min-p --repeat-penalty
--freq-penalty --presence-penalty --seed --greedy`); `--gen N` does raw token generation,
`--info` prints the config. Constrained decoding mirrors qwen3 too
(`--json-schema/--lark/--regex` on `--chat`/`--repl`, `-Dllguidance=true` builds).

---

## DiffusionGemma 26B-A4B (block text-diffusion) — `zig build diffusion-gemma`

Weights: `diffusiongemma-26B-A4B-it-Q6_K.gguf` from
[`unsloth/diffusiongemma-26B-A4B-it-GGUF`](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF)
(Gemma Terms of Use apply).

Not autoregressive: it denoises 256-token canvases with the entropy-bound sampler
(defaults from the model: ≤48 steps, temperature 0.8→0.4, bound 0.1) and commits blocks
autoregressively. Expect a few seconds per denoising step on the 26B MoE; simple answers
converge in ~7 steps.

On a TTY, chat replies **denoise live inline** — the streaming equivalent for diffusion:
the reply repaints in place where it belongs in the transcript, with not-yet-accepted
tokens faint (they "crystallize" as the sampler converges) and a dim trailing status line
(`… step 3/48 · accepted 201/256 · H̄ 0.522`); when the block finalizes, the clean text
simply remains, followed by a dim stats trailer. `--no-visual` disables it, `--visual`
forces it when piped, `--visual-interval N` redraws every Nth step.

```sh
# Chat (Gemma 4 turn template; reply denoises inline on a TTY)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "Why is the sky blue? Answer in two sentences." --max 256 --seed 42

# --experts=borrow: zero-copy MoE expert load. The Q6_K model otherwise x4-packs
# ~20 GB on load (this is the slow/swappy default on memory-tight boxes); borrow
# loads it in ~2.5 s at ~half the RSS. Default pack favors peak throughput.
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "Why is the sky blue? Answer in two sentences." --experts=borrow

# Multi-turn interactive REPL (context carries across turns; empty line or Ctrl-D quits;
# like llama.cpp -cnv, each turn re-encodes the full history)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --repl --system "Answer tersely."

# Longer multi-block generation (each 256-token canvas is re-encoded into the KV cache)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "Write a 400-word short story about a lighthouse keeper." --max 512 --seed 7

# Tune the entropy-bound sampler (higher bound = more tokens accepted per step = faster/riskier)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "..." --steps 32 --entropy-bound 0.2 --t-max 0.9 --t-min 0.4

# Disable self-conditioning (slightly cheaper, usually worse convergence)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "..." --no-sc

# Raw-token block generation (no chat template)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --gen 256 2,818,7217,7412

# Config/tokenizer info without loading weights
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf --info

# Logit-parity harness vs llama.cpp PR #24423's llama-diffusion-gemma-eval
# (prompt ids + EXACTLY canvas_length=256 canvas ids; dumps raw f32 [256, vocab])
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --eval 2,651,235 --canvas <256-comma-separated-ids> \
  --logits-out /tmp/dg.bin --compare-logits /tmp/oracle.bin
# add --sc-logits prev.bin to feed a previous step's logits as self-conditioning (temp_inv=1)
```

Other knobs: `--confidence F` and `--stability N` (the adaptive stop), `--system "..."`,
`--think`.

---

## Qwen3.5 0.8B (Gated-DeltaNet hybrid) — `zig build qwen35`

Weights: `Qwen3.5-0.8B-Q8_0.gguf` from
[`unsloth/Qwen3.5-0.8B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF).

Loader/parity harness (streaming decode is in the module; the CLI is minimal):

```sh
zig build qwen35 -Doptimize=ReleaseFast -- models/Qwen3.5-0.8B-Q8_0.gguf
zig build qwen35 -Doptimize=ReleaseFast -- models/Qwen3.5-0.8B-Q8_0.gguf --info
zig build qwen35 -Doptimize=ReleaseFast -- models/Qwen3.5-0.8B-Q8_0.gguf --linear-scan chunked
```

---

## OpenAI-compatible LM server — `zig build lmserve`

One process serves one model behind `POST /v1/chat/completions` and the
stateless `POST /v1/responses` (plus `GET /v1/models`, `GET /health`), with
SSE streaming in both dialects. The GGUF's `general.architecture` picks the
backend (qwen3/qwen3moe/gemma4/diffusion-gemma); `--nanochat <dir>` serves a
nanochat checkpoint. Point any OpenAI client at `http://host:port/v1`. See
`docs/LMSERVER.md` for the exact API mapping and design.

```sh
# Serve Qwen3 with JSON-schema/regex/Lark constrained output enabled
zig build lmserve -Dllguidance=true -Doptimize=ReleaseFast -- \
  models/Qwen3-0.6B-Q8_0.gguf --port 8080

# Talk to it with any OpenAI client (SSE streaming: add "stream": true)
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"Give me facts about Paris."}],
  "response_format": {"type":"json_schema","json_schema":{"name":"city","schema":{
    "type":"object","properties":{"city":{"type":"string","maxLength":30},
    "population":{"type":"integer","maximum":99999999}},
    "required":["city","population"],"additionalProperties":false}}}}'

# Gemma 4 MoE (zero-copy expert load) / nanochat checkpoint dir
zig build lmserve -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf --experts=borrow
zig build lmserve -Doptimize=ReleaseFast -- --nanochat runs/sft
```

Flags: `--host --port --ctx --api-key --queue --conns --experts=borrow
--nanochat` (`--help` lists them). Requests are accepted concurrently and
generated sequentially (one inference worker; the queue bounds admission —
overflow gets 429). Reasoning is off by default; clients enable it per
request via `reasoning_effort` (qwen3 routes `<think>` text to
`reasoning_content`).

---

## OmniVoice TTS (MaskGIT, voice cloning/design) — `zig build omnivoice`

Weights: `omnivoice-base-{F32,BF16,Q8_0,Q4_K_M}.gguf` +
`omnivoice-tokenizer-{F32,BF16,Q8_0,Q4_K_M}.gguf` from
[`Serveurperso/OmniVoice-GGUF`](https://huggingface.co/Serveurperso/OmniVoice-GGUF)
(GGUF conversions of [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice); the
tokenizer GGUF is the Higgs Audio v2 codec — HuBERT + DAC + RVQ). All 16 base×tokenizer
combinations work. Q8_0 base + F32 tokenizer is the recommended pairing (fastest sane
quality; the C++ reference's own default). Avoid the reference's BF16 base for speed
comparisons — ggml's CPU BF16 matmul path is pathologically slow (~470 s vs our ~14 s
per clip); Fucina runs BF16 at F32-class speed via resident-bf16 weights.

```sh
# voice design (six attribute categories; see examples/omnivoice/voicedesign.zig for the vocab)
echo "Hello from Fucina." | zig build omnivoice -Doptimize=ReleaseFast -- tts \
  --model models/omnivoice/omnivoice-base-Q8_0.gguf \
  --codec models/omnivoice/omnivoice-tokenizer-F32.gguf \
  --lang English --instruct "female, young adult, high pitch" --seed 42 -o out.wav

# voice cloning (a 5-15 s reference wav + its transcript; 48 kHz input is resampled)
echo "Text to speak in the cloned voice." | zig build omnivoice -Doptimize=ReleaseFast -- tts \
  --model models/omnivoice/omnivoice-base-Q8_0.gguf \
  --codec models/omnivoice/omnivoice-tokenizer-F32.gguf \
  --lang English --ref-wav reference.wav --ref-text ref-transcript.txt --seed 42 -o out.wav

# auto voice + chunked long-form (chunk 0 locks the speaker for later chunks)
zig build omnivoice -Doptimize=ReleaseFast -- tts --model ... --codec ... \
  --lang English -o out.wav < long-text.txt

# speaker playback (--playback <idx> picks a device; `devices` lists them)
zig build omnivoice -Doptimize=ReleaseFast -- tts --model ... --codec ... \
  --lang English --play < text.txt

# stream WAV to stdout as chunks finish ('-o -'); --stream-by-line = one WAV
# header per input line
zig build omnivoice -Doptimize=ReleaseFast -- tts --model ... --codec ... \
  --lang English -o - < long-text.txt | ffplay -autoexit -nodisp -i -

# codec round-trip (wav -> .rvq -> wav) and parity oracles
zig build omnivoice -Doptimize=ReleaseFast -- codec --model models/omnivoice/omnivoice-tokenizer-F32.gguf -i clip.wav
zig build omnivoice -Doptimize=ReleaseFast -- codec --model models/omnivoice/omnivoice-tokenizer-F32.gguf -i clip.rvq
zig build omnivoice -- tts --model ... --codec ... --maskgit-test --duration 3 --lang English -o tokens.bin < text.txt
```

Determinism: greedy (`--maskgit-test`) and seeded runs are token-exact vs the C++
reference ([omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp)) on F32;
quantized bases produce equally-valid but *different* token streams — quantizing the
backbone perturbs activations enough to flip sampling decisions, and the two
implementations diverge the same way from each other as two quant levels do. CPU perf
(M1 Max, cool chip, shipped defaults): 2.3–4.6× faster than omnivoice.cpp's CPU backend
across F32/Q8_0/Q4_K_M design+clone runs.

---

## Parakeet ASR (NeMo FastConformer) — `zig build parakeet`

Weights: [`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf) —
one flat repo with every supported NVIDIA NeMo Parakeet model × quantization
(f16/q8_0/q6_k/q5_k/q4_k). Start with `tdt_ctc-110m-f16.gguf` (267 MB hybrid, fast) or
`tdt-0.6b-v3-f16.gguf` (1.44 GB, multilingual). Input is 16 kHz mono WAV.

```sh
mkdir -p models/parakeet
hf download mudler/parakeet-cpp-gguf tdt_ctc-110m-f16.gguf --local-dir models/parakeet

# Transcribe a WAV (clean transcript on stdout)
zig build parakeet -Doptimize=ReleaseFast -- --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --audio clip.wav --transcribe

# JSON output / per-word timestamps (offline decode only)
zig build parakeet -Doptimize=ReleaseFast -- --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --audio clip.wav --transcribe --json --timestamps

# Batch a manifest (one audio path per line)
zig build parakeet -Doptimize=ReleaseFast -- --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --manifest files.txt

# Streaming pipeline (cache-aware chunked encode; use a streaming-capable model)
zig build parakeet -Doptimize=ReleaseFast -- --model models/parakeet/tdt_ctc-110m-f16.gguf \
  --audio clip.wav --stream

# Live microphone (build the capture backend in with -Dparakeet-mic)
zig build parakeet -Dparakeet-mic -Doptimize=ReleaseFast -- \
  --model models/parakeet/tdt_ctc-110m-f16.gguf --mic

# Benchmarks: best-of-N offline timing / streaming RTF + first-token latency
zig build parakeet -Doptimize=ReleaseFast -- --model ... --audio clip.wav --transcribe --bench-reps 5
zig build parakeet -Doptimize=ReleaseFast -- --model ... --audio clip.wav --stream-bench --bench-reps 5
```

Other flags: `--decoder tdt|ctc` (hybrid models), `--lang XX` (multilingual
prompt-conditioned models; default auto), `--threads N`, `--mic-sim` (feed `--audio`
through the incremental mic driver), `--compare <stage> <dump>` + `--tol F` (stage-level
parity vs parakeet.cpp dumps), `--f32-cache`, `--fast-mel`. Running with only `--model`
prints the config + tensor summary.

Honest perf note: transcription is parity-checked against parakeet.cpp/NeMo, but
parakeet.cpp is still faster on CPU — on the 110m hybrid, Fucina full-transcribe
RTF ≈ 0.034 vs parakeet.cpp ≈ 0.021 (M1 Max), a ~1.6× gap. See `BENCHMARK.md`.

---

## LocateAnything-3B (open-vocabulary detection) — `zig build locate-anything`

Give it an image and a text prompt; it returns labeled boxes. Port of
[mudler/locate-anything.cpp](https://github.com/mudler/locate-anything.cpp)
(NVIDIA `LocateAnything-3B`: MoonViT vision tower + MLP projector + Qwen2.5-3B;
detection happens in token space via coordinate tokens). The GGUF uses that
port's `locateanything.*` schema — build it with the reference converter
(`scripts/convert_locateanything_to_gguf.py` in the reference repo, f32), or
quantize the LM matmuls with its CLI (`quantize ... q8_0`); Fucina reads both.

```sh
# Detect: labeled boxes as JSON (byte-compatible with the reference CLI)
zig build locate-anything -Doptimize=ReleaseFast -- detect \
  --model models/locate-anything-f32.gguf --input scene.png \
  --prompt 'Locate all the instances that matches the following description: cat</c>remote.' \
  --mode hybrid --output boxes.json

# Decode modes mirror upstream generation_mode: hybrid (parallel box decoding
# with AR fallback, default), slow (pure autoregressive), fast (MTP-only).
# --no-early-stop reproduces the reference's full uncapped token stream.

# Parity gates vs a reference dump (tools/ref-patches/la_dump.cpp): exit 1 on any failure
zig build locate-anything -Doptimize=ReleaseSafe -- compare \
  --model models/locate-anything-f32.gguf --dump dumps/fixture_dump.gguf \
  --image parity_image.png --prompt '...' --stage all
```

PNG input only (the pure-Zig reader; PNG decode is lossless so pixels are
byte-identical to the reference's stb path).

---

## Neural Amp Modeler — `zig build nam`

Runs standard `.nam` guitar-amp captures (WaveNet/LSTM/ConvNet/Linear, upstream format
0.5.0–0.7.x). Get profiles from [TONE3000 (formerly ToneHunt)](https://tonehunt.org/) —
free community captures, no account required — or train your own from a reamp pair.
Profiles in `./nam-profiles` (or `$FUCINA_NAM_PROFILES`) show up in the interactive menu
(`zig build nam` with no arguments).

```sh
# Offline render (matches upstream tools/render); --ir appends a cabinet IR
zig build nam -Doptimize=ReleaseFast -- render "profile.nam" input.wav output.wav --ir cab.wav

# Live playing through the profile (realtime; `devices` lists audio/MIDI devices)
zig build nam -Doptimize=ReleaseFast -- live "profile.nam" --ir cab.wav
zig build nam -- devices

# Inspect / benchmark a profile
zig build nam -- inspect "profile.nam"
zig build nam -Doptimize=ReleaseFast -- bench "profile.nam" --blocksize 128

# Train a WaveNet from an input/reamp WAV pair, then validate (reports ESR)
zig build nam -Doptimize=ReleaseFast -- train --input in.wav --output reamp.wav \
  --out model.nam --spec standard
zig build nam -- validate model.nam --input in.wav --output reamp.wav

# Lossless .nam <-> GGUF interchange
zig build nam -- export-gguf model.nam model.gguf
```

`zig build nam -- --help` lists the full command set (profile capture, chains, MIDI
mapping, loopback latency test).

---

## Fine-tune → merge → serve loop

Uses any Qwen3 dense GGUF from the sources above.

```sh
# LoRA fine-tune a Qwen3 GGUF on CPU (built-in pirate-style SFT set; ~0.9 s/step on 0.6B)
zig build finetune -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-Q4_K_S.gguf \
  --steps 30 --rank 8 --alpha 16 --save /tmp/qwen3-lora

# Useful extras: --lr F  --seq-max N  --checkpoint-layers (activation checkpointing)
#                --load PATH (resume)  --seed N  --verify-grads (gradient-evidence audit)
#                --accum-steps N (gradient accumulation windows, exact token-weighted; recipe in TRAINING.md §4)
#                --state-dtype f32|bf16 (bf16 optimizer moments; TRAINING.md §3/§8)
#                --data PATH.jsonl  --shuffle  --data-seed N (SFT data via src/llm/data.zig;
#                resume CONTINUES the data order — loader state lives in trainer_state.json)

# Merge the adapters into dense weights (merge needs an f32/f16/bf16 base — a quantized
# base errors; see the f16 note in "Getting the weights". Merge and --dtype are separate
# passes by design: one combined pass would chain-requantize.)
zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf models/Qwen3-0.6B-f16.gguf \
  --out tuned-f16.gguf --adapters /tmp/qwen3-lora --alpha 16

# Quantize the merged model in a second pass (runs in llama.cpp too)
zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf tuned-f16.gguf \
  --out tuned.gguf --dtype q8_0

# Serve the result (CLI chat, or over HTTP — see "OpenAI-compatible LM server")
zig build qwen3 -Doptimize=ReleaseFast -- tuned.gguf --chat "Who are you?"
zig build lmserve -Doptimize=ReleaseFast -- tuned.gguf --port 8080
```

`export-gguf` also re-emits/transcodes without adapters
(`--dtype f16|bf16|f32|q8_0|q4_k|q5_k|q6_k|verbatim`) and PTQTP-quantizes
tensor-at-a-time with `--ptqtp[=K]` — models far bigger than RAM stream
from the source mmap into `<name>.ptqtp0..K-1` trit-plane tensors that the
family loaders pair-detect (`--ptqtp-include/--ptqtp-exclude` name filters,
`--dry-run` plan preview; docs/PTQTP.md).

### Gradient-free variant: evolution strategies

`es-finetune` is finetune's ES-at-scale twin (same data/checkpoint plumbing,
no backward pass, no optimizer state — see `TRAINING.md` §13):

```sh
# ES fine-tune the LoRA q/v adapters (quantized base is fine — forward passes only).
# --reward acc = bounded token-accuracy + 0.1*exp(-CE) composite (recommended: cheap AND stable);
# --reward nll = raw negative CE (loss-comparable with finetune, but unbounded — degrades on
#                long runs unless paired with --norm centered_ranks and --save-every selection);
# --reward rule = R1-style rule reward on greedy generations (generation-bound, slower).
zig build es-finetune -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-Q4_K_S.gguf \
  --reward acc --iterations 100 --population 8 --batch 5 --sigma 0.02 --save-every 25 \
  --save /tmp/qwen3-es

# Full-parameter ES (the paper's setting): every resident float weight of the model.
# Needs an f32/f16/bf16 GGUF — quantized blocks cannot take gaussian noise.
zig build es-finetune -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --mode full --reward nll --iterations 10 --population 4 --batch 2 --sigma 0.001

# Useful extras: --alpha F (default sigma/2)  --noise iid|correlated  --antithetic  --norm z_score|centered_ranks
#                --restore regenerate|snapshot  --anchor-decay l1|l2 --anchor-lambda F (AWD anti-drift)
#                --max-new N + --format-prefix/--format-suffix (rule reward)  --load DIR (resume:
#                (seed, es_iteration) regenerate the population stream)  --data/--shuffle/--data-seed
```

The lora checkpoint is the same adapters.safetensors finetune writes, so the
merge → quantize → serve loop above applies unchanged.

### Cartridges: train a corpus into a reusable KV prefix — `zig build cartridge`

Cartridges (arXiv 2506.06266; design record `docs/CARTRIDGES.md`) compress a
document into the KV cache of a virtual p-token prefix by in-process
self-study distillation — the model interviews itself about the corpus, and
the teacher-with-context distills into a small trainable cache that is then
served like any cached prompt. Needs an f32/f16/bf16 GGUF (gradients flow
through the frozen weights; see the f16 note in "Getting the weights").

The exact demo below uses the repository's own `README.md` as the corpus and
`Qwen3-0.6B-f16.gguf`. Measured on an M1 Max: the training command runs
**~4.5 minutes end-to-end** — 3.3 min of self-study (32 conversations at
≈6 s each: chunk sampling, both bot generations, teacher scoring, backward)
plus model load, capture init, save, and the three serve answers (the
5.4k-token ICL prefill is the slow tail). The CLI prints its own timing
per step and a `self-study: ... s/conversation` summary.

```sh
# 1. Acceptance gate (~10 s): an UNTRAINED cartridge built from the model's
#    own K/V rows for the first 256 corpus tokens must score the next 128
#    tokens exactly like the real prefill. Expected output ends with:
#      prefill-equivalence over 128 suffix tokens: max |dlogit| 0.000000, ... greedy agreement 128/128
#      PASS: untrained corpus-init cartridge is behaviorally identical to the real prefill
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --corpus README.md --p 256 --suffix-max 128 --equiv

# 2. Self-study training + three-way comparison (~4.5 min on an M1 Max):
#    32 synthesized conversations (8 optimizer steps x 4-conversation
#    accumulation), teacher top-20 targets, Adam lr 2e-3 (the default;
#    docs/CARTRIDGES.md explains why the paper's 2e-2 needs its 65k-token
#    batches). Prints per-step distill loss, the held-out loss before/after,
#    saves the cartridge, then answers --ask three ways. Expected: the
#    [cartridge, 256 KV rows] and [ICL, ~5.3k KV rows] answers agree
#    ("Fucina is a CPU-first tensor/autograd runtime and LLM inference
#    engine written in pure Zig 0.16.") while [bare model] hallucinates.
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --corpus README.md --p 256 --steps 8 --chunk-min 256 --chunk-max 512 \
  --max-q 64 --max-a 160 --seed 7 --save /tmp/fucina-cartridge-readme.safetensors \
  --draft-ref --ask "What is Fucina, in one sentence?"

# 3. Serve the saved cartridge later — geometry is recovered from the
#    safetensors header; --corpus is optional (enables the ICL column).
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --load /tmp/fucina-cartridge-readme.safetensors \
  --ask "What backends and hardware does Fucina support?" --corpus README.md

#    Training with --draft-ref embeds the corpus token ids in the artifact
#    (8 bytes/token), making it self-contained for speculative serving:
#    --spec-serve builds the corpus suffix automaton ONCE at load (~1 ms per
#    5k tokens) and the corpus drafts the answer — no --corpus needed, and
#    nothing is constructed per generation call. Output is byte-identical to
#    plain decoding (lossless verification); +12-16% tok/s on long
#    corpus-grounded answers. Artifacts without the entry fall back to
#    drafting from --corpus.
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --load /tmp/fucina-cartridge-readme.safetensors --spec-serve \
  --ask "What backends and hardware does Fucina support?"

# 4. Multi-document corpora: --corpus repeats, and a directory takes its
#    top-level .md files in sorted order; every file is prefixed with a
#    "# Document: <path>" header and each training chunk carries a one-line
#    provenance description. The repo's own documentation (19 files,
#    ~328k tokens — far beyond a sane prefill) is a corpus in one flag set;
#    the ICL comparison column truncates at --icl-max (default 4096) tokens.
#    Measured (M1 Max): ~8 min end-to-end — 64 conversations in 6.6 min at
#    the SAME ~6 s/conversation as the 5k-token corpus. The --equiv gate is
#    bitwise on this corpus too.
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --corpus README.md --corpus AGENTS.md --corpus docs \
  --p 512 --steps 16 --chunk-min 256 --chunk-max 512 --max-q 64 --max-a 160 \
  --seed 7 --save /tmp/fucina-cartridge-docs.safetensors \
  --ask "What is Fucina, in one sentence?"

# 5. Full-coverage runs: checkpoint every N steps (atomic, same --save
#    path) and resume from a checkpoint (rows only; Adam moments restart).
#    2048 conversations over the full docs ~ 4-5 h on an M1 Max, ~15 min
#    per checkpoint interval below. GPU note: with the batched pipeline
#    -Dgpu=metal is ~1.2x faster per conversation on an M1 Max at 0.6B
#    (~2x at 1.7B with --gen-batch 16) — see docs/CARTRIDGES.md
#    "Acceleration".
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --corpus README.md --corpus AGENTS.md --corpus docs \
  --p 1024 --steps 512 --accum 4 --max-a 192 --seed 7 \
  --save cartridge-full.safetensors --save-every 32 --draft-ref \
  --ask "What is Fucina, in one sentence?"
# resume after an interruption:
#   ... --resume cartridge-full.safetensors --steps 256 --save cartridge-full.safetensors --save-every 32

# 6. Serve it over HTTP: every conversation of the OpenAI-compatible server
#    preloads the cartridge — requests answer from the corpus with ZERO
#    corpus tokens in the prompt, and cross-request KV reuse (cached_tokens)
#    operates on the real tokens past the prefix. See docs/LMSERVER.md.
zig build lmserve -Doptimize=ReleaseFast -- models/Qwen3-0.6B-f16.gguf \
  --port 8080 --cartridge /tmp/fucina-cartridge-readme.safetensors
# curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
#   -d '{"model":"m","messages":[{"role":"user","content":"What is Fucina, in one sentence?"}]}'

# Useful extras: --p N (prefix rows)  --frozen N (attention-sink rows, default 1)
#                --steps/--accum/--lr (optimizer)  --chunk-min/--chunk-max (corpus spans)
#                --top-k N (teacher entries/token)  --max-q/--max-a (bot budgets)  --seed N
#                --icl-max N (ICL comparison context cap)  --save-every N  --resume PATH
#                --draft-ref (embed the corpus ids in the artifact for --spec-serve)
#                --gen-batch N (generation stream width, decoupled from --accum;
#                a multiple of it — wider batches amortize the decode weight stream)
#                --spec-b / --spec-serve (lossless speculative decoding: self-study bot B /
#                corpus-drafted serving; see docs/CARTRIDGES.md)
#                --no-pack (flat-memory per-conversation backward; the default packs the
#                accumulation group into one forward/backward — gradient-identical, and the
#                group's generations run as lockstep batched streams either way: measured
#                1.57x conversations/s over the sequential pipeline)
```

Generation is sampled (bot A, temperature 0.6), so conversations — and the
trained rows — vary with `--seed`; the answers above are what the pinned
seed produces. Per-conversation cost is INDEPENDENT of corpus size (only
the sampled chunk enters the teacher's context — the 328k-token corpus
trains at the same ~6 s/conversation as the 5k-token one); corpus size
instead sets the COVERAGE budget: 64 conversations sample ~7% of the full
documentation, so knowledge quality on big corpora scales with conversation
count (the paper's regime is tens of thousands). These demo budgets show
the mechanism, not the ceiling.

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
