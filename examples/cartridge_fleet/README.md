# Cartridge fleet — one cartridge per document

Train one KV-prefix cartridge PER DOCUMENT under a RAM/disk budget, then
pick and compose cartridges per query with an in-process retriever.

Cartridges at Scale (arXiv 2606.04557; design record
[`docs/CARTRIDGES.md`](../../docs/CARTRIDGES.md) §"Cartridges at Scale")
trains one cartridge per document under a RAM/disk budget, jointly so they
compose at serve time, and selects cartridges per query with an in-process
cosine retriever (embeddings come from the serving model itself — no
external retrieval stack). qwen3 and gemma4 GGUFs (gemma routes to the flat
per-conversation backward and per-conversation teacher passes, like the
base CLI; expert blocks borrow zero-copy); qwen3 needs an f32/f16/bf16 GGUF
like the base CLI. The single-corpus CLI — training mechanics, acceptance
gate, speculative serving — is
[`examples/cartridge`](../cartridge/README.md).

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

## Corpus layout

Each `--docs` argument is one document (a text file, up to 16 MiB) or a
directory: its top-level `.md` files become one document each, in sorted
name order (other files and subdirectories are ignored). Manifest document
names are the paths exactly as passed, so run from the repo root and keep
the spellings stable across runs. The `--equiv` gate concatenates every
document's tokens and needs at least `2*p + 2` of them.

## Walkthrough

### 1. Composition acceptance gate (~10 s)

Two cartridges built from ONE capture of the first 512 corpus tokens (part
B holds rows at positions 256..511) must reproduce the real prefill over
the next 128 tokens.

```sh
zig build cartridge-fleet -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --docs README.md --equiv --p 256 --suffix-max 128
```

Expected output ends with:

```
composed prefill-equivalence over 128 suffix tokens (2 x p = 512 rows):
  max |dlogit| 0.000000, ... greedy agreement 128/128
PASS: the two-part composition is behaviorally identical to the real prefill
```

### 2. Train a 4-document fleet (~10 min on an M1 Max, ~5.5 s/conversation)

Every doc gets a corpus-init cartridge on disk, at most `--budget 3` stay
resident, and 24 rounds x 4 conversations of mixed-visibility self-study
run with rotation every 6 rounds (isolated and co-loaded x2/x3 rounds;
rotation yields uniform coverage — per-doc step counts 8/8/8/9 with only 3
of 4 ever resident). Ends by embedding the 88 retrieval chunks through the
model (~44 s) and saving `fleet.json` + per-doc safetensors/FZT1 +
`index.safetensors`.

```sh
zig build cartridge-fleet -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --docs README.md --docs docs/TERNARY.md --docs docs/PTQTP.md --docs docs/SPECULATIVE.md \
  --fleet /tmp/fleet-demo --p 256 --budget 3 --rotate-every 6 --rounds 24 --accum 4 --seed 7
```

### 3. Serve with cartridge-RAG

The question embeds through the model, the cosine top chunks pick
documents, and the selected cartridges compose ahead of the question
(mmap-loaded, ~0.9 s to first answer for one 256-row cartridge, ~2.5-3.3 s
for two). Measured with the fleet above: "What is Fucina, in one sentence?"
selects README.md and answers "a CPU-first tensor/autograd runtime and LLM
inference engine written in pure Zig 0.16..." while `[bare model]` answers
"a traditional Italian dessert"; the speculative-decoding question composes
SPECULATIVE+PTQTP and describes batched verification against the committed
stream.

```sh
zig build cartridge-fleet -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --fleet /tmp/fleet-demo --ask "What is Fucina, in one sentence?" --rag-docs 1
zig build cartridge-fleet -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --fleet /tmp/fleet-demo --ask "In speculative decoding, how are draft tokens verified?" --rag-docs 2
```

### 4. Oracle selection and resume

`--oracle NAME` bypasses retrieval (the paper's oracle arm); `--resume`
reopens a fleet to keep training (rows + Adam moments continue exactly
where they left off — evict/reload is bit-identical); `--rounds 0 --resume`
rebuilds only the retrieval index. `--resume` is a bare flag here — the
fleet path comes from `--fleet`.

```sh
zig build cartridge-fleet -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --fleet /tmp/fleet-demo --ask "..." --oracle docs/TERNARY.md
```

Resume runs pass the SAME `--docs` list (names match the manifest by exact
string; order may differ) — training modes always reload the document
texts:

```sh
zig build cartridge-fleet -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --docs README.md --docs docs/TERNARY.md --docs docs/PTQTP.md --docs docs/SPECULATIVE.md \
  --fleet /tmp/fleet-demo --resume --budget 3 --rotate-every 6 --rounds 24 --accum 4 --seed 7
```

The same command with `--rounds 0` rebuilds only the retrieval index.

### 5. Serve the fleet over HTTP

Each request's user messages pick documents through the fleet's cosine
index and the selected cartridges compose as the conversation's prefix;
follow-up turns stick to the selection their conversation started with and
report `cached_tokens` through it (`docs/LMSERVER.md`; see
[`examples/lmserve`](../lmserve/README.md)). Measured: the README question
answers the canonical sentence from a 22-token prompt; an interleaved
follow-up reused 86/87 prompt tokens warm.

```sh
zig build lmserve -Doptimize=ReleaseFast -- models/Qwen3-0.6B-f16.gguf \
  --port 8080 --fleet /tmp/fleet-demo --kv-slots 4
# curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
#   -d '{"model":"m","messages":[{"role":"user","content":"What is Fucina, in one sentence?"}]}'
```

Server selection knobs: `--rag-docs K` (documents composed per request,
default 2) and `--rag-chunks N` (cosine top-N chunks scanned, default 8).
`--rag-adaptive` lets a CONTINUING conversation switch knowledge base when
a document outside its selection decisively out-scores it (`--rag-margin`,
default 0.05) under the contextual query — the switch rebuilds the prefix
and re-prefills (`cached_tokens` = 0 that turn); default is fully sticky
(selection pinned at conversation start), and a NEW conversation always
re-retrieves. Size `--ctx` to include rag_docs x p prefix rows. `--fleet`
excludes `--cartridge`/`--kv-cache-dir`.

## Fleet directory layout

`--fleet DIR` holds:

| file | contents |
| --- | --- |
| `fleet.json` | manifest: `p`, `frozen_prefix`, `embed_chunk`, `embed_dim`, rounds so far, per-document name/token/step counts and file names |
| `doc-NNN.safetensors` | document NNN's cartridge rows |
| `doc-NNN.fza` | its Adam-moment snapshot (FZT1); a missing or rows-only artifact reloads with fresh moments |
| `index.safetensors` | the centered, normalized chunk-embedding cosine index |

Evict/reload through these files is bit-identical to staying resident.

## gemma fleets

gemma GGUFs run the same modes: the composed `--equiv` gate judges greedy
flips against the model's own shape-sensitivity envelope on quantized MoE,
init/index/serving run end to end on 26B, and `lmserve --fleet` serves
gemma fleets (MoE GGUFs need `--experts=borrow`). 26B TRAINING runs at
~210 s/conversation and needs >=128 GB of RAM (the backward transient peaks
at 58-118 GB — `docs/CARTRIDGES.md` "gemma4 fleets"); `--rounds 0` builds a
served-ready corpus-init fleet + index:

```sh
zig build cartridge-fleet -Doptimize=ReleaseFast -- --model models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --docs README.md --docs docs/TERNARY.md --fleet /tmp/fleet-gemma \
  --p 64 --budget 2 --rounds 0 --embed-chunk 512
```

## Training knobs

| flag | meaning |
| --- | --- |
| `--budget B` | resident cartridges |
| `--rotate-every R` | rotation interval (rounds) |
| `--evict-frac F` | fraction evicted per rotation |
| `--warmup W` | per-cartridge lr warm-up steps |
| `--p-iso F` | isolation probability |
| `--distract-max K` | co-loaded distractors |
| `--rounds` / `--accum` / `--lr` / `--p` / `--chunk-min` / `--chunk-max` / `--max-q` / `--max-a` / `--seed` | self-study budgets (the base cartridge CLI's demo defaults) |
| `--embed-chunk N` | retrieval chunk tokens |
| `--rag-docs` / `--rag-chunks` | selection |
| `--no-pack` | flat-memory per-conversation backward |
| `--checkpoint` | per-layer recompute, qwen3: halves training peak RSS, byte-identical results; forces isolated visibility in fleet runs |

Defaults follow the paper's recipe, re-calibrated to the CLI's demo
batches (`docs/CARTRIDGES.md` "Learning rate vs batch size" — the paper's
Adam lr 2e-2 is calibrated to 32×2048-token packed batches; the CLI ships
2e-3): `--budget 4`, `--rotate-every 10`,
`--evict-frac 0.5`, `--warmup 8`, `--p-iso 0.75`, `--distract-max 3`,
`--rounds 20`, `--accum 4`, `--lr 2e-3`, `--embed-chunk 256`,
`--rag-docs 2`, `--rag-chunks 8`.

The CLI also accepts `--frozen N` and `--top-k N` (same meaning as the
single-cartridge CLI) and `--suffix-max N` (the `--equiv` gate's suffix
length).

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, `-Dcpu` when cross-compiling),
GPU offload, MoE expert streaming, global thread/BLAS knobs, and
`-Dllguidance` constrained decoding are shared machinery — see
[`docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md). Model-specific
note: gemma MoE fleets served through `lmserve --fleet` need
`--experts=borrow`.
