# Qwen3.5 — Gated-DeltaNet hybrid loader/parity harness

Runner for the `qwen35` architecture: full-attention blocks interleaved with
DeltaNet-linear blocks (conv1d + DeltaNet scan + multi-section RoPE). The CLI
is minimal by design — it loads a GGUF, prints the derived config and per-kind
block counts, and runs the hybrid forward pass with a top-5 logit readout.
Streaming decode is in the `llm.qwen35` module; chat serving goes through
[lmserve](../lmserve/README.md).

## Getting the model

Weights are not part of this repository. `Qwen3.5-0.8B-Q8_0.gguf` comes from
[`unsloth/Qwen3.5-0.8B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)
(also mirrored by lmstudio-community and bartowski):

```sh
mkdir -p models
hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q8_0.gguf --local-dir models
```

## CLI

```sh
zig build qwen35 -Doptimize=ReleaseFast -- models/Qwen3.5-0.8B-Q8_0.gguf
zig build qwen35 -Doptimize=ReleaseFast -- models/Qwen3.5-0.8B-Q8_0.gguf --info
zig build qwen35 -Doptimize=ReleaseFast -- models/Qwen3.5-0.8B-Q8_0.gguf --linear-scan chunked
```

Full usage:
`zig build qwen35 -- <model.gguf> [<token-ids>] [--info] [--decode] [--profile] [--bench R [--gen N]] [--logits-out PATH] [--linear-scan chunked|recurrent]`

| flag | meaning |
| --- | --- |
| `<token-ids>` | comma-separated prompt token ids (default `9707,11,1879`) |
| `--info` | print the GGUF-derived config and exit (answers from metadata alone, no weight load) |
| `--decode` | incremental-decode equivalence check: prefill one token, decode the rest through the streaming cache (KV + recurrent state), compare final logits to the whole-sequence forward (argmax match + mean abs diff) |
| `--profile` | per-stage timing breakdown of the forward pass |
| `--bench R` | prefill (pp) / decode (tg) throughput sweep, best-of-`R` |
| `--gen N` | decode-step count for `--bench` (default 128) |
| `--logits-out PATH` | dump the forward-pass f32 logits for external parity checks |
| `--linear-scan chunked\|recurrent` | DeltaNet scan implementation (default `chunked`) |

## Ternary-Bonsai-27B (Prism ML, ternary Q2_0 g128)

A ternarized Qwen3.6-27B on the same `qwen35` architecture (64 blocks,
16 full-attention + 48 DeltaNet-linear, 48 v-heads over 16 q/k heads,
262K-token context) with every projection, the embeddings and the LM head
in the Q2_0 ternary container — ~7.2 GB on disk, runs comfortably on a
laptop. Weights (Apache-2.0):

```sh
hf download prism-ml/Ternary-Bonsai-27B-gguf Ternary-Bonsai-27B-Q2_0.gguf --local-dir models
```

```sh
# loader / parity harness
zig build qwen35 -Doptimize=ReleaseFast -- models/Ternary-Bonsai-27B-Q2_0.gguf --info

# serve it (chat + reasoning channel + constrained output; see ../lmserve/README.md)
zig build lmserve -Dllguidance=true -Doptimize=ReleaseFast -- \
  models/Ternary-Bonsai-27B-Q2_0.gguf --port 8080
```

The chat template is ChatML with the Qwen3.6 thinking prefill; reasoning is
off by default and enabled per request via `reasoning_effort`. Sampling
defaults come from the GGUF's `general.sampling.*` keys.

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, `-Dcpu=...` when cross-compiling),
global thread/BLAS knobs, GPU offload, and `-Dllguidance` constrained-decoding
usage are documented once in
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). This harness takes only
the flags listed above; serving knobs belong to lmserve.
