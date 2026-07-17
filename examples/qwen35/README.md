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

## Parity oracle

`--logits-out` writes the final-position logits as raw little-endian f32
(`n_vocab` values), computed from explicit token ids so tokenizer
differences cannot masquerade as model bugs. The reference half is
[`tools/llama_logits.cpp`](../../tools/llama_logits.cpp) — same format,
same position — compiled against the pinned llama.cpp checkout
(methodology in [docs/BENCHMARK.md](../../docs/BENCHMARK.md)):

```sh
tools/fetch_refs.sh --build llama.cpp   # clone/pin + CPU build of the oracle
c++ -O2 -std=c++17 tools/llama_logits.cpp \
  -I refs/llama.cpp/include -I refs/llama.cpp/ggml/include \
  -L refs/llama.cpp/build-cpu/bin -lllama \
  -Wl,-rpath,"$PWD/refs/llama.cpp/build-cpu/bin" -o /tmp/llama_logits
```

Run both sides on the same ids; acceptance is top-token alignment (the
reference prints its argmax to stderr, the harness prints top-5 — a small
mean |Δ| between independent implementations is expected):

```sh
/tmp/llama_logits models/Qwen3.5-0.8B-Q8_0.gguf 9707,11,1879 /tmp/ref.bin
zig build qwen35 -Doptimize=ReleaseFast -- models/Qwen3.5-0.8B-Q8_0.gguf \
  9707,11,1879 --logits-out /tmp/f.bin
```

The harness takes raw token ids only; ids for arbitrary text come from the
reference tokenizer:

```sh
refs/llama.cpp/build-cpu/bin/llama-tokenize \
  -m models/Qwen3.5-0.8B-Q8_0.gguf -p "Hello, world" --ids
```

For Ternary-Bonsai-27B the oracle is Prism ML's llama.cpp fork (the Q2_0
g128 container is its addition — see
[docs/THIRD-PARTY-NOTICES.md](../../docs/THIRD-PARTY-NOTICES.md)); its
`llama-tokenize` is the token-ID gate. Same CPU-build flags as the
mainline build in `tools/fetch_refs.sh`:

```sh
tools/fetch_refs.sh prism-llama.cpp
cmake -S refs/prism-llama.cpp -B refs/prism-llama.cpp/build-cpu \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF -DLLAMA_CURL=OFF
cmake --build refs/prism-llama.cpp/build-cpu -j --target llama-tokenize
```

then compile `llama_logits` with the `refs/prism-llama.cpp` include/lib
paths and dump/compare as above.

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
