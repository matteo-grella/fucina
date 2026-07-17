# DeepSeek V4 Flash 284B-A13B — `zig build deepseek4`

Greedy completion over the CSA/HCA trunk with streamed experts, native MTP
speculative decoding from a sidecar GGUF, and two parity oracles against
the upstream reference.

This is one of the three DeepSeek-family runners — the siblings are
[DeepSeek V2/V3 (`deepseek2`)](../deepseek2/README.md) and
[GLM-4.5 (`glm4moe`)](../glm4moe/README.md). All three share the
streamed-expert machinery and accept `--moe-stream`/`--moe-cache-mb`
([Streaming MoE experts from disk](../../docs/RUNNING-MODELS.md#streaming-moe-experts-from-disk-out-of-core-models-bigger-than-ram)).

The DwarfStar-class trunk: hyper-connections (4 residual streams mixed by a
Sinkhorn-normalized combine), MQA over a single 512-dim FP8-simulated KV row
with per-head sink logits, streaming compressors with an FP4/Hadamard
indexer (top-512 row selection), sqrt-softplus routing with hash-routed
early layers, and a grouped low-rank output projection. The 164.6 GB Q4K
release decodes on a 64 GB machine with `--moe-stream` (measured 1.5–3.6
tok/s warm at a 20 GB expert budget). `--chat` renders the reference
template (thinking disabled: BOS, user marker, prompt, assistant marker,
closed think block).

The port follows Salvatore Sanfilippo's ds4 reference implementation
([antirez/ds4](https://github.com/antirez/ds4), MIT), which is also the
parity oracle — `docs/THIRD-PARTY-NOTICES.md` records the lineage.

## Getting the model

Model weights are not part of this repository. Weights:
[huggingface.co/antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf)
— mixed-precision single-GGUF exports of DeepSeek-V4-Flash (arch tag
`deepseek4`), including the MTP sidecar GGUF.

## Run

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
```

Flags (first positional argument = model GGUF, required):

| flag | meaning |
| --- | --- |
| `--prompt "..."` / `--prompt=...` | prompt text (default `The capital of France is`) |
| `--prompt-file=PATH` | read the prompt from a file (≤ 16 MiB) |
| `--gen N` / `--gen=N` | greedy tokens to generate, default 16 |
| `--chat` | render the reference chat template (thinking disabled) instead of raw completion |
| `--prefill-chunk=N` | batched-prefill chunk size, default 128; `1` = sequential |
| `--mtp=PATH` | MTP sidecar GGUF for native speculative decoding |
| `--mtp-depth=N` | draft depth, default 1, caps at 8 |
| `--index-probe` | decode-time selection-overlap probe across CSA layers; measures the exact path, so mutually exclusive with `--index-share` |
| `--index-share=N` | cross-layer indexer reuse: every Nth Full CSA layer computes its selection, the layers between reuse it — approximate by design, calibrate with the probe first |
| `--vectors=DIR` / `--vectors-max-prompt=N` | official-vector regression, see below |
| `--golden=PATH` | local-golden logit oracle, see below |
| `--moe-stream` / `--moe-cache-mb=N` | streamed experts — see *Shared knobs* |

## Parity oracles

Both replay fixtures shipped in the upstream `ds4` checkout and exit
nonzero on failure. Fetch the pinned checkout first — the fixtures are
consumed in place:

```sh
tools/fetch_refs.sh ds4

# Official-vector regression. The default --vectors-max-prompt=256 runs
# the three short fixtures and skips the two ~3.4–3.8k-token ones; raise
# it to run all five (--prefill-chunk sizes the batched prefill):
zig build deepseek4 -Doptimize=ReleaseFast -- <model.gguf> --moe-stream \
  --vectors=refs/ds4/tests/test-vectors/official --vectors-max-prompt=4096

# Implementation-level logit oracle: replay the upstream local-golden
# fixture (top-64 ids + raw logits at a 4096-token frontier) with the
# upstream pass thresholds:
zig build deepseek4 -Doptimize=ReleaseFast -- <model.gguf> --moe-stream \
  --golden=refs/ds4/tests/test-vectors/local-golden.vec
```

`--vectors` runs every `*.official.json` fixture with the reference chat
rendering and greedy decoding and compares the continuation against the
official API's step by step, on concatenated bytes (a different token
boundary with identical text still matches). A vector fails only when it
diverges on the very first step: quantized weights legitimately drift a few
steps in, but step 0 disagreeing means the forward is wrong.

`--golden` prefills the fixture's frontier prompt tokens (mode `text`:
plain BPE, no BOS) and compares the frontier logits against the recorded
top-64 with the upstream thresholds: top-1 exact, top-5 ≥ 4, top-20 ≥ 15,
top-64 ≥ 40, top-20 max |Δ| ≤ 8.

The fixture's prompt file resolves relative to the checkout root (two
levels above the `.vec`), so point `--golden` at the file inside
`refs/ds4/` — a copied `.vec` loses its prompt (`error: FileNotFound`).

The checkout is never built here. Do not run its `make cpu` on macOS —
it can kernel-panic the VM system (`tools/fetch_refs.sh` records this).

## Shared knobs

MoE expert streaming, GPU offload (`-Dgpu=metal`/`-Dgpu=cuda`), global
thread/BLAS knobs and the ReleaseFast/`-Dcpu` build discipline are shared
machinery — see [docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md).
Of the streaming knob set this runner parses only `--moe-stream` and
`--moe-cache-mb`.
