# fucina-run (`zig build run`)

One GGUF runner over the architecture registry: `zig build run -- <model.gguf>`
sniffs `general.architecture`, resolves the family through `models.registry`,
and drives it through the decoder contract. Registered architectures:
qwen3, qwen3moe, gemma4, qwen35, qwen35moe, inkling, deepseek2, deepseek4,
glm4moe. llama.cpp split GGUFs (`-00001-of-0000N`) load transparently from
the first part.

The family harnesses with their own surface keep their own apps:
[`zig build qwen3`](../qwen3/README.md) (speculative decode, batch decode,
SHINE adapters), [`zig build deepseek4`](../deepseek4/README.md) (MTP
sidecar, cascade speculation, official-vector parity),
[`zig build diffusion-gemma`](../diffusion_gemma/README.md) (block
diffusion), [`zig build gemma4`](../../examples/gemma4/README.md) and
[`zig build qwen35`](../../examples/qwen35/README.md) (logit-parity
harnesses), and [`zig build lmserve`](../lmserve/README.md) (HTTP serving).

## Run

```sh
# Completion (greedy by default; sampling past --temp 0)
zig build run -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" --gen 32

# Template chat / REPL (qwen3, qwen3moe, gemma4)
zig build run -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --repl
```

Prints load/prefill/decode timings, the generated token ids, and the
completion text.

## Common flags

First positional argument = model GGUF (required); a second positional
comma-separated token-id list replaces prompt encoding (parity input).

| flag | meaning |
| --- | --- |
| `--prompt "..."` / `--prompt=...` | prompt text (default `The capital of France is`) |
| `--prompt-file=PATH` | read the prompt from a file (up to 64 MiB) |
| `--gen N` / `--gen=N` | tokens to generate, default 32 (0 with a parity dump flag) |
| `--chat "msg"` / `--repl` / `--system "..."` / `--no-think` | template chat (one-shot / multi-turn) via the generic `Conversation`; inkling routes through its wire-format engine |
| `--temp F`, `--top-k=N`, `--top-p=F`, `--min-p=F`, `--repeat-penalty=F`, `--seed=N` | sampling; `--temp 0` (default) is greedy |
| `--ctx=N` | KV capacity override (default `max(2048, prompt+gen+8)`; 4096 for chat) |
| `--prefill-chunk=N` | batched-prefill chunk size, default 64; `1` restores the sequential S=1 path |
| `--nll-file=PATH` / `--nll-tokens=N` | teacher-forced NLL/perplexity over a text file, then exit |
| `--tokenize FILE` | encode a text file, one token id per line (the `llama-tokenize` parity rung); no weights needed |
| `--logits-out PATH` | raw f32 dump of the last-position logits after prefill |
| `--compare-logits PATH` / `--max-abs F` | exit-code gate vs a reference logits dump (default gate 1e-4) |
| `--step1` | token-at-a-time prefill (decode-vs-batch parity rung) |
| `--bench R` | best-of-R pp/tg throughput (with `--image`/`--audio`: tower preprocess+encode) |
| `--threads N` | worker-thread override |
| `--info` | print the GGUF's architecture string and exit |

Streamed experts (the shared `models.moe_stream_cli` set: `--moe-stream`,
`--moe-cache-mb=N`, mirrors, L2 tier, trace) plus `--moe-pilot`,
`--moe-cache-route`, `--moe-route-j=N`, `--moe-route-m=N`, `--moe-pin-mb=N`,
`--moe-no-learn`, and `--moe-cache-slots=N` apply where the family loader
supports streaming (deepseek2, glm4moe, deepseek4, qwen35moe); see
[Streaming MoE experts from disk](../../docs/RUNNING-MODELS.md#streaming-moe-experts-from-disk-out-of-core-models-bigger-than-ram).

## DeepSeek V2/V3 (MLA): arch `deepseek2`

Multi-head latent attention with the compressed KV cache as the default
(576 floats per token per layer; `--mla=full` selects the reconstructing
path, byte-identical output) and weight absorption folding `kv_b` into the
query/value sides. Covers V2-Lite (softmax router), V3-style checkpoints
such as Moonlight-16B-A3B (sigmoid no-aux router, q-LoRA, MLA-native GGUF
layout), and GLM `glm-dsa` checkpoints. When the vocab defines
`[gMASK]`/`<sop>` the prompt opens with them instead of BOS; GLM trunks
degenerate without that opening. No download source is pinned for this
family; any V2/V3-family GGUF works.

```sh
zig build run -Doptimize=ReleaseFast -- \
  models/DeepSeek-V2-Lite-Chat.Q8_0.gguf --prompt "..." --gen 64
```

| flag | meaning |
| --- | --- |
| `--mla=latent\|full` | `latent` (default) decodes from the compressed KV cache; `full` reconstructs heads (byte-identical output) |
| `--dsa` | load the DSA lightning-indexer tensors (V3.2 / glm-dsa files) and attend sparsely past `indexer_top_k` positions (the trained behavior of those checkpoints) |
| `--dsa-top-k=N` | selection-threshold override so the sparse path fires within a short prompt; selection semantics unchanged |
| `--index-probe` | decode-time selection-overlap probe across DSA layers; mutually exclusive with `--index-share` |
| `--index-share=N` | cross-layer indexer reuse: every Nth DSA layer computes its selection, the layers between reuse it; approximate by design, calibrate with the probe first |
| `--moe-experts=N` | inference-time truncation of the routed-expert count; dropped experts are never fetched, gate weights renormalize |
| `--moe-top-p=F` | dynamic expert drop: keep routed experts covering fraction F of the gate mass (deterministic) |
| `--moe-skip-miss=F` | dynamic expert drop: skip sub-threshold-weight experts only when they would cost a disk read (cache-state dependent output) |

## GLM-4.5 family: arch `glm4moe`

V3-style MoE trunk plus the model's own `nextn` multi-token-prediction
layer: `--mtp[=depth]` drafts with the MTP head through the library's
`MtpDraftSource` and verifies with one batched trunk step in the shared
`SpeculativeDecoder` loop; only greedy-matching prefixes commit, so
output is lossless (byte-identical to plain greedy; the same prompt and
`--gen` with and without `--mtp` must print identical `generated ids` and
`text` lines). Bare `--mtp` is depth 2; values above 8 clamp to 8 (the
kernel-pinned verify's losslessness bound). On a model without a `nextn`
layer `--mtp` is ignored with a notice. The canonical GLM `[gMASK]<sop>`
opening is added automatically.

Weights:
[`unsloth/GLM-4.5-Air-GGUF`](https://huggingface.co/unsloth/GLM-4.5-Air-GGUF):
the `Q6_K/` folder holds a two-part split (~99 GB total) whose
conversion keeps the `nextn` layer:

```sh
hf download unsloth/GLM-4.5-Air-GGUF \
  Q6_K/GLM-4.5-Air-Q6_K-00001-of-00002.gguf \
  Q6_K/GLM-4.5-Air-Q6_K-00002-of-00002.gguf \
  --local-dir models/glm45-air
mv models/glm45-air/Q6_K/*.gguf models/glm45-air/

zig build run -Doptimize=ReleaseFast -- \
  models/glm45-air/GLM-4.5-Air-Q6_K-00001-of-00002.gguf \
  --prompt "..." --gen 64 --mtp --moe-stream --moe-cache-mb=20480
```

At ~99 GB the weights outsize a 64 GB machine's RAM; the run above
streams the experts from disk. With `--mtp` the decode line reports tokens
per forward and the draft-acceptance rate.

## Inkling 975B-A41B: arch `inkling`

The thinkingmachines/Inkling architecture: 66 layers alternating local
(512-token window) and global attention with a banded content-dependent
relative-position bias instead of RoPE, per-layer short causal convolutions
on four sites, a 256-expert top-6 sigmoid-routed MoE with 2 shared experts
as routing sinks, log-N attention scaling, and muP logit scaling.
Reference: llama.cpp PR #25731, pinned on the `llama.cpp-inkling` ref
(`tools/fetch_refs.sh`; the oracle build recipe and the two
`tools/ref-patches/llama.cpp-inkling-*.patch` files the tower parity dumps
need are in that script's comments). Prompts encode raw (no BOS), matching
the oracle.

Decoder GGUFs:
[`unsloth/inkling-GGUF`](https://huggingface.co/unsloth/inkling-GGUF):
the smallest quant (UD-IQ1_S) is 270 GB in 7 split files; pass the first
part. The image and audio towers are a separate small file (183 MB) at the
root of the same repo:

```sh
mkdir -p models
hf download unsloth/inkling-GGUF mmproj-BF16.gguf --local-dir models
```

```sh
# Parity harness (ids in, logits/generation out):
zig build run -Doptimize=ReleaseFast -- <model.gguf> --tokenize file.txt
zig build run -Doptimize=ReleaseFast -- <model.gguf> 13225,2375 \
  --logits-out ours.bin --compare-logits ref.bin --max-abs 1e-4
zig build run -Doptimize=ReleaseFast -- <model.gguf> --prompt "..." --gen 64

# Chat (typed-block wire format; --repl multi-turn, --no-think skips
# reasoning, --system sets the system message). Sampler-driven (--temp).
zig build run -Doptimize=ReleaseFast -- <model.gguf> --chat "Hi!" --system "..."
zig build run -Doptimize=ReleaseFast -- <model.gguf> --repl --temp 0.7

# Multimodal (one <__media__> marker; PNG images, WAV audio):
zig build run -Doptimize=ReleaseFast -- <model.gguf> --mmproj <mmproj.gguf> \
  --image photo.png --prompt "Describe this: <__media__> in short." --gen 64
zig build run -Doptimize=ReleaseFast -- <model.gguf> --mmproj <mmproj.gguf> \
  --audio clip.wav --prompt "Transcribe: <__media__>" --gen 64 --embd-out t.bin
```

Multimodal input goes through `--mmproj <mmproj.gguf>` with either
`--image` or `--audio` (mutually exclusive; one media file per run): the
prompt carries one `<__media__>` marker where the media embeddings enter
the decoder. `--image` accepts 8-bit non-interlaced PNG (grayscale, RGB, or
RGBA); palette and 16-bit PNGs are rejected and JPEG is not supported.
`--audio` accepts WAV with PCM 16/24/32-bit int or 32-bit float samples
(incl. WAVE_FORMAT_EXTENSIBLE) at any sample rate and channel count;
multi-channel audio is downmixed by averaging and resampled to the tower's
16 kHz mono when the source rate differs.

Tower-only smoke run (no decoder needed): `--embd-out` without `--gen`,
`--logits-out`, or `--compare-logits`, and `--bench R` with a media flag,
return after the tower encode, so the mmproj GGUF itself satisfies the
positional `<model.gguf>` argument:

```sh
zig build run -Doptimize=ReleaseFast -- models/mmproj-BF16.gguf \
  --mmproj models/mmproj-BF16.gguf --image photo.png \
  --prompt "<__media__>" --embd-out img_embd.bin
zig build run -Doptimize=ReleaseFast -- models/mmproj-BF16.gguf \
  --mmproj models/mmproj-BF16.gguf --audio clip.wav \
  --prompt "<__media__>" --bench 3
```

Full multimodal generation additionally needs a decoder whose hidden size
matches the mmproj embedding width (`error.MmprojWidthMismatch` otherwise).

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, `-Dcpu`), GPU offload, and the
global thread/BLAS knobs are documented in
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md).
