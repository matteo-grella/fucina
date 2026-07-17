# lmserve — OpenAI-compatible LM server

One process serves one model behind `POST /v1/chat/completions` and the
stateless `POST /v1/responses` (plus `GET /v1/models`, `GET /health`), with
SSE streaming in both dialects. Point any OpenAI client at
`http://host:port/v1`.

The GGUF's `general.architecture` picks the backend: `qwen3`, `qwen3moe`,
`qwen35` (Qwen3.5 / Ternary-Bonsai), `gemma4`, `diffusion-gemma`, `inkling`;
`--nanochat <dir>` serves a nanochat checkpoint (`model.safetensors` +
`tokenizer.bin`). This README is the getting-started face;
**[`docs/LMSERVER.md`](../../docs/LMSERVER.md) is the full design doc** —
exact API mapping tables (what is honored, rejected, ignored per dialect),
streaming contracts, scheduler design, and the KV-reuse machinery.

## Getting a model

Weights are not part of this repository. Any GGUF of a supported family
works — the full download matrix per family is in
[`docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md#getting-the-weights).
The two used below:

```sh
mkdir -p models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models
hf download unsloth/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-UD-Q6_K.gguf --local-dir models
```

**Gemma license.** Gemma-family weights (Gemma 4, DiffusionGemma) are
distributed under Google's Gemma Terms of Use. The `google/…` originals on
Hugging Face are gated behind accepting those terms; the unsloth GGUF
conversions were not gated at the time of writing, but the terms still apply
to the weights either way.

The other backends:

- Qwen3 MoE 30B-A3B, Qwen3.5-0.8B, and DiffusionGemma: rows with
  `hf download` commands in the same
  [matrix](../../docs/RUNNING-MODELS.md#getting-the-weights).
- Ternary-Bonsai-27B (qwen35 architecture, Apache-2.0):
  `hf download prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0.gguf --local-dir models`
  — serving notes in [`../qwen35/README.md`](../qwen35/README.md).
- Inkling: no practically sized GGUF exists — public releases run to
  hundreds of GB (unsloth/inkling-GGUF UD-IQ1_S: 270 GB). The backend is exercised via
  the parity harness in [`../inkling/README.md`](../inkling/README.md).
- nanochat: nothing to download — `--nanochat` serves a checkpoint dir
  (`model.safetensors` + `tokenizer.bin`) trained with the pipeline in
  [`../nanochat/README.md`](../nanochat/README.md).

## Running

```sh
# Serve Qwen3 with JSON-schema/regex/Lark constrained output enabled
zig build lmserve -Dllguidance=true -Doptimize=ReleaseFast -- \
  models/Qwen3-0.6B-Q8_0.gguf --port 8080

# Gemma 4 MoE (zero-copy expert load) / nanochat checkpoint dir
zig build lmserve -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf --experts=borrow
zig build lmserve -Doptimize=ReleaseFast -- --nanochat runs/sft
```

## Talking to it

Any OpenAI client works; with curl (SSE streaming: add `"stream": true`):

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"Hi!"}]}'
```

Constrained output (JSON-schema shown; regex and Lark grammars too — needs
the `-Dllguidance=true` build):

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"Give me facts about Paris."}],
  "response_format": {"type":"json_schema","json_schema":{"name":"city","schema":{
    "type":"object","properties":{"city":{"type":"string","maxLength":30},
    "population":{"type":"integer","maximum":99999999}},
    "required":["city","population"],"additionalProperties":false}}}}'
```

The stateless Responses dialect takes `input` (a string or typed message
items) plus `instructions`:

```sh
curl -s http://127.0.0.1:8080/v1/responses -H 'Content-Type: application/json' -d '{
  "input": "Hi!", "instructions": "Answer in one sentence."}'
```

Reasoning, per request (`reasoning.effort` in the responses dialect;
rejected when the model has no toggleable reasoning channel):

```sh
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"What is 17*23?"}], "reasoning_effort": "low"}'
```

```sh
curl -s http://127.0.0.1:8080/v1/models   # the served model id
curl -s http://127.0.0.1:8080/health
```

The request's `model` field is accepted and ignored — one process serves one
model; `GET /v1/models` reports its id.

## Flags

`--help` lists them all:

| flag | meaning |
| --- | --- |
| `--host H` | bind address (default 127.0.0.1) |
| `--port N` | port (default 8080) |
| `--ctx N` | per-request context budget in tokens (default 4096) |
| `--api-key K` | require `Authorization: Bearer K` |
| `--queue N` | max queued requests before 429 (default 16) |
| `--conns N` | max concurrent connections (default 32) |
| `--experts=borrow` | zero-copy MoE expert load (gemma4/diffusion-gemma) |
| `--nanochat DIR` | serve a nanochat checkpoint dir |
| `--kv-slots N` | resident KV-reuse slots (default 1); each holds a full `--ctx` cache, so extra slots cost real memory but keep interleaved conversations warm |
| `--kv-cache-dir D` | spill evicted slots to sidecar files under `D` and restore them on prefix match (GGUF chat backends) |
| `--kv-disk-slots M` | max sidecar files under `--kv-cache-dir` (default 8) |
| `--cartridge F` | preload a trained KV-prefix cartridge into every conversation (see below) |
| `--fleet DIR` | serve a per-document cartridge fleet (see below) |
| `--rag-docs` `--rag-chunks` `--rag-adaptive` `--rag-margin` | fleet retrieval knobs (see below) |

## Queue and reasoning semantics

Requests are accepted concurrently and generated sequentially (one inference
worker; the queue bounds admission — overflow gets 429). Reasoning is off by
default; clients enable it per request via `reasoning_effort` (chat) or
`reasoning.effort` (responses) — `"none"`/`"minimal"` disable,
`"low"`/`"medium"`/`"high"`/`"xhigh"`/`"default"` enable (rejected when the
model has no toggleable reasoning channel). qwen3 routes `<think>` text to
`reasoning_content`.

## Cartridge serving

The qwen3/gemma4 backends serve trained KV-prefix cartridges — corpus
knowledge with zero prompt tokens
([`docs/CARTRIDGES.md`](../../docs/CARTRIDGES.md)):

- `--cartridge F` preloads one cartridge (safetensors from `zig build
  cartridge`, see [`../cartridge/README.md`](../cartridge/README.md)) into
  every conversation; composes with the slot pool and the `--kv-cache-dir`
  disk tier.
- `--fleet DIR` serves a per-document fleet (from `zig build cartridge-fleet`,
  see [`../cartridge_fleet/README.md`](../cartridge_fleet/README.md)): each
  request's last user message picks `--rag-docs` cartridges via the fleet's
  cosine index (`--rag-chunks` chunks scanned) and they compose as the
  conversation's prefix. Selection is sticky per conversation;
  `--rag-adaptive` lets follow-up turns switch knowledge base when an outside
  document decisively out-scores the selection (margin `--rag-margin`,
  default 0.05). gemma4 MoE GGUFs need `--experts=borrow`; excludes
  `--cartridge` and `--kv-cache-dir`.

Both flags expect artifacts trained against the same model GGUF being
served — KV geometry is probed at startup, so a mismatched file fails
there, not mid-request. Size `--ctx` to include the prefix (fleet:
`rag_docs × p` rows on top of the conversation). Training recipes:
[`docs/CARTRIDGES.md`](../../docs/CARTRIDGES.md); full serving semantics:
[`docs/LMSERVER.md`](../../docs/LMSERVER.md).

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, `-Dcpu` when cross-compiling),
GPU offload, thread/BLAS settings, and `-Dllguidance` constrained-decoding
usage are shared across runners — see
[`../../docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md). This server
does not take `--moe-stream`; its MoE knob is `--experts=borrow` above.
