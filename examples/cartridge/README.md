# Cartridge — train a corpus into a reusable KV prefix

Distill a document into the KV cache of a virtual p-token prefix, then serve
it like any cached prompt — corpus-grounded answers with zero corpus tokens
in the prompt.

Cartridges (arXiv 2506.06266; design record
[`docs/CARTRIDGES.md`](../../docs/CARTRIDGES.md)) compress a document into
the KV cache of a virtual p-token prefix by in-process self-study
distillation — the model interviews itself about the corpus, and the
teacher-with-context distills into a small trainable cache that is then
served like any cached prompt. Needs an f32/f16/bf16 GGUF (gradients flow
through the frozen weights). The multi-document sibling — one cartridge per
document with in-process retrieval — is
[`examples/cartridge_fleet`](../cartridge_fleet/README.md).

The exact demo below uses the repository's own `README.md` as the corpus and
`Qwen3-0.6B-f16.gguf`. Measured on an M1 Max: the training command runs
**~4.5 minutes end-to-end** — 3.3 min of self-study (32 conversations at
≈6 s each: chunk sampling, both bot generations, teacher scoring, backward)
plus model load, capture init, save, and the three serve answers (the
5.4k-token ICL prefill is the slow tail). The CLI prints its own timing
per step and a `self-study: ... s/conversation` summary.

## Getting the weights

Weights are not part of this repository. The walkthrough uses
`Qwen3-0.6B-f16.gguf`; bf16 Qwen3-0.6B GGUFs come from
[`bartowski/Qwen_Qwen3-0.6B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF)
or [`unsloth/Qwen3-0.6B-GGUF`](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF)
(see the weights table in [`docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md)).
If your source only ships bf16, transcode one locally:

```sh
zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf <src>.gguf \
  --out models/Qwen3-0.6B-f16.gguf --dtype f16
```

The gemma arm uses the quantized MoE GGUF:

```sh
mkdir -p models
hf download unsloth/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-UD-Q6_K.gguf --local-dir models
```

Gemma-family weights are distributed under Google's Gemma Terms of Use. The
`google/…` originals on Hugging Face are gated behind accepting those terms;
the unsloth GGUF conversions were not gated at the time of writing, but the
terms still apply to the weights either way.

## Walkthrough

### 1. Acceptance gate (~10 s)

An UNTRAINED cartridge built from the model's own K/V rows for the first 256
corpus tokens must score the next 128 tokens exactly like the real prefill.

```sh
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --corpus README.md --p 256 --suffix-max 128 --equiv
```

Expected output ends with:

```
prefill-equivalence over 128 suffix tokens: max |dlogit| 0.000000, ... greedy agreement 128/128
PASS: untrained corpus-init cartridge is behaviorally identical to the real prefill
```

### 2. Self-study training + save (~3.5 min on an M1 Max)

32 synthesized conversations (8 optimizer steps x 4-conversation
accumulation), teacher top-20 targets, Adam lr 2e-3 (the default;
`docs/CARTRIDGES.md` explains why the paper's 2e-2 needs its 65k-token
batches). Prints per-step distill loss and the held-out loss before/after,
then saves.

```sh
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --corpus README.md --p 256 --steps 8 --chunk-min 256 --chunk-max 512 \
  --max-q 64 --max-a 160 --seed 7 --draft-ref \
  --save /tmp/fucina-cartridge-readme.safetensors
```

### 3. Serve the saved cartridge

Geometry is recovered from the safetensors header; `--corpus` is optional
(enables the ICL column). Expected: the `[cartridge, 256 KV rows]` and
`[ICL, ~5.3k KV rows]` answers agree ("Fucina is a CPU-first tensor/autograd
runtime and LLM inference engine written in pure Zig 0.16.") while
`[bare model]` hallucinates.

```sh
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --load /tmp/fucina-cartridge-readme.safetensors \
  --ask "What is Fucina, in one sentence?" --corpus README.md
```

Training with `--draft-ref` embeds the corpus token ids in the artifact
(8 bytes/token), making it self-contained for speculative serving:
`--spec-serve` builds the corpus suffix automaton ONCE at load (~1 ms per
5k tokens) and the corpus drafts the answer — no `--corpus` needed, and
nothing is constructed per generation call. Output is byte-identical to
plain decoding (lossless verification); +12-16% tok/s on long
corpus-grounded answers. Artifacts without the entry fall back to drafting
from `--corpus`.

```sh
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --load /tmp/fucina-cartridge-readme.safetensors --spec-serve \
  --ask "What backends and hardware does Fucina support?"
```

### 4. Multi-document corpora

`--corpus` repeats, and a directory takes its top-level `.md` files in
sorted order; every file is prefixed with a `# Document: <path>` header and
each training chunk carries a one-line provenance description. The repo's
own documentation (19 files, ~328k tokens — far beyond a sane prefill) is a
corpus in one flag set; the ICL comparison column truncates at `--icl-max`
(default 4096) tokens. Measured (M1 Max): ~8 min end-to-end — 64
conversations in 6.6 min at the SAME ~6 s/conversation as the 5k-token
corpus. The `--equiv` gate is bitwise on this corpus too.

```sh
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --corpus README.md --corpus AGENTS.md --corpus docs \
  --p 512 --steps 16 --chunk-min 256 --chunk-max 512 --max-q 64 --max-a 160 \
  --seed 7 --save /tmp/fucina-cartridge-docs.safetensors
```

### 5. Full-coverage runs

Checkpoint every N steps (atomic, same `--save` path) and resume from a
checkpoint (rows only; Adam moments restart). 2048 conversations over the
full docs ~ 4-5 h on an M1 Max, ~15 min per checkpoint interval below. GPU
note: with the batched pipeline `-Dgpu=metal` is ~1.2x faster per
conversation on an M1 Max at 0.6B (~2x at 1.7B with `--gen-batch 16`) — see
`docs/CARTRIDGES.md` "Acceleration".

```sh
zig build cartridge -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --corpus README.md --corpus AGENTS.md --corpus docs \
  --p 1024 --steps 512 --accum 4 --max-a 192 --seed 7 \
  --save cartridge-full.safetensors --save-every 32 --draft-ref
# resume after an interruption:
#   ... --resume cartridge-full.safetensors --steps 256 --save cartridge-full.safetensors --save-every 32
```

### 6. Serve it over HTTP

Every conversation of the OpenAI-compatible server preloads the cartridge —
requests answer from the corpus with ZERO corpus tokens in the prompt, and
cross-request KV reuse (`cached_tokens`) operates on the real tokens past
the prefix. See [`examples/lmserve`](../lmserve/README.md) and
`docs/LMSERVER.md`.

```sh
zig build lmserve -Doptimize=ReleaseFast -- models/Qwen3-0.6B-f16.gguf \
  --port 8080 --cartridge /tmp/fucina-cartridge-readme.safetensors
# curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
#   -d '{"model":"m","messages":[{"role":"user","content":"What is Fucina, in one sentence?"}]}'
```

## gemma routing

gemma GGUFs (dense or MoE) route to the gemma4 trainer arm; `--equiv` runs
the acceptance gate with the model's shape-sensitivity envelope printed
first (quantized-MoE stacks are not GEMM-shape-invariant — see
`docs/CARTRIDGES.md` "gemma4"). Train via the gemma4 trainer API; serve via
`lmserve --cartridge`.

```sh
zig build cartridge -Doptimize=ReleaseFast -- --model models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --corpus README.md --equiv --p 64 --suffix-max 32
```

## Knobs

| flag | meaning |
| --- | --- |
| `--p N` | prefix rows |
| `--frozen N` | attention-sink rows, default 1 |
| `--steps` / `--accum` / `--lr` | optimizer |
| `--chunk-min` / `--chunk-max` | corpus spans |
| `--top-k N` | teacher entries/token |
| `--max-q` / `--max-a` | bot budgets |
| `--seed N` | self-study sampling seed (see below) |
| `--icl-max N` | ICL comparison context cap |
| `--save-every N` | checkpoint interval |
| `--resume PATH` | resume from a checkpoint (rows only; Adam moments restart) |
| `--draft-ref` | embed the corpus ids in the artifact for `--spec-serve` |
| `--gen-batch N` | generation stream width, decoupled from `--accum`; a multiple of it — wider batches amortize the decode weight stream |
| `--spec-b` / `--spec-serve` | lossless speculative decoding: self-study bot B / corpus-drafted serving; see `docs/CARTRIDGES.md` |
| `--checkpoint` | recompute-in-backward per layer: measured 1.7B peak RSS 15.7 -> 7.9 GB with a byte-identical trained artifact, and faster; qwen3 single-cartridge path |
| `--no-pack` | flat-memory per-conversation backward; the default packs the accumulation group into one forward/backward — gradient-identical, and the group's generations run as lockstep batched streams either way: measured 1.57x conversations/s over the sequential pipeline |

## Seed and coverage

Generation is sampled (bot A, temperature 0.6), so conversations — and the
trained rows — vary with `--seed`; the answers above are what the pinned
seed produces. Per-conversation cost is INDEPENDENT of corpus size (only
the sampled chunk enters the teacher's context — the 328k-token corpus
trains at the same ~6 s/conversation as the 5k-token one); corpus size
instead sets the COVERAGE budget: 64 conversations sample ~7% of the full
documentation, so knowledge quality on big corpora scales with conversation
count (the paper's regime is tens of thousands). These demo budgets show
the mechanism, not the ceiling.

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, `-Dcpu` when cross-compiling),
GPU offload (`-Dgpu=metal` / `-Dgpu=cuda`), MoE expert streaming, global
thread/BLAS knobs, and `-Dllguidance` constrained decoding are shared
machinery — see [`docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md).
Model-specific note: GPU builds accelerate self-study training itself
(prefill-shaped GEMMs offload; step 5 above and `docs/CARTRIDGES.md`
"Acceleration").
