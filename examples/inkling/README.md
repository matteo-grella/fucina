# Inkling 975B-A41B — hybrid rel-bias attention + MoE

Runs the thinkingmachines/Inkling architecture from GGUF: text generation,
chat, and multimodal input through the ported image and audio towers.

66 layers alternate local (512-token window) and global attention with a
banded content-dependent relative-position bias instead of RoPE, plus
per-layer short causal convolutions on four sites (k-proj, v-proj, attention
output, FFN output), a 256-expert top-6 sigmoid-routed MoE whose 2 shared
experts participate in the routing softmax as sinks, log-N attention scaling
past 128k tokens, and muP logit scaling. Reference: llama.cpp PR #25731,
pinned on the `llama.cpp-inkling` ref (`tools/fetch_refs.sh`).

The full release has NOT been run here — the smallest public GGUF
(unsloth/inkling-GGUF UD-IQ1_S) is 270 GB. Parity is closed against the
pinned oracle on a synthetic full-architecture checkpoint covering every
code path (both layer kinds, per-layer KV-head counts, all conv sites, MoE
routing, log-N scaling, padded-vocab masking): tokenizer token-ID-exact on
14 adversarial fixtures plus multi-kilobyte real files against the REAL
201k-token Inkling tokenizer; last-position logits max-abs < 1e-6 over
prompt lengths 5–200 on batch prefill and token-at-a-time decode; and
128-token greedy generation id-exact vs the oracle. The mmproj towers are
ported too — the hMLP vision stem (byte-exact fixed-point Lanczos
resampling, 40x40 patchify, one patch = one decoder token) and the dMel
audio tower (one 50 ms frame = one token) — with tiny-pair e2e logits
< 1e-6 and 24-token greedy id-exact for image and audio across adversarial
geometries, and REAL mmproj-BF16 towers at audio max-abs 3.8e-6 (exact
tier) / vision min cosine 0.9999977 (bf16-weights tier; ggml rounds GEMM
activations to bf16, fucina accumulates f32). The CPU path (SIMD attention
fan-out over the worker team, per-expert BLAS-shaped MoE prefill GEMMs,
last-position-only 201k-wide unembed, batched tower lookups) measured ahead
of the pinned llama.cpp build on prefill, decode, and both real-mmproj
tower encodes (like-for-like `--bench` vs llama-bench/mtmd, 8 threads).

## Weights

Decoder GGUFs:
[unsloth/inkling-GGUF](https://huggingface.co/unsloth/inkling-GGUF). The
smallest quant (UD-IQ1_S) is 270 GB in 7 split files
(`UD-IQ1_S/inkling-UD-IQ1_S-00001-of-00007.gguf` …) — pass the first
`-00001-of-0000N` part and the remaining parts are mapped automatically
(split loading is covered in
[../../docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md)).

The image and audio towers are a separate small file (183 MB) at the root
of the same repo:

```sh
mkdir -p models
hf download unsloth/inkling-GGUF mmproj-BF16.gguf --local-dir models
```

(`hf` comes from `pip install -U huggingface_hub` —
[../../docs/RUNNING-MODELS.md#getting-the-weights](../../docs/RUNNING-MODELS.md#getting-the-weights).)

## Commands

```sh
# Parity harness (ids in, logits/generation out):
zig build inkling -Doptimize=ReleaseFast -- <model.gguf> --tokenize file.txt
zig build inkling -Doptimize=ReleaseFast -- <model.gguf> 13225,2375 \
  --logits-out ours.bin --compare-logits ref.bin --max-abs 1e-4
zig build inkling -Doptimize=ReleaseFast -- <model.gguf> --prompt "..." --gen 64

# Chat (typed-block wire format; --repl multi-turn, --no-think skips reasoning,
# --system sets the system message). Sampler-driven (--temp).
zig build inkling -Doptimize=ReleaseFast -- <model.gguf> --chat "Hi!" [--system "..."]
zig build inkling -Doptimize=ReleaseFast -- <model.gguf> --repl --temp 0.7

# Multimodal (one <__media__> marker; PNG images, WAV audio):
zig build inkling -Doptimize=ReleaseFast -- <model.gguf> --mmproj <mmproj.gguf> \
  --image photo.png --prompt "Describe this: <__media__> in short." --gen 64
zig build inkling -Doptimize=ReleaseFast -- <model.gguf> --mmproj <mmproj.gguf> \
  --audio clip.wav --prompt "Transcribe: <__media__>" --gen 64 [--embd-out t.bin]
```

Multimodal input goes through `--mmproj <mmproj.gguf>` with either `--image`
or `--audio` (mutually exclusive — one media file per run); the prompt
carries one `<__media__>` marker where the media embeddings enter the
decoder. `--image` accepts 8-bit non-interlaced PNG (grayscale, RGB, or
RGBA); palette and 16-bit PNGs are rejected and JPEG is not supported.
`--audio` accepts WAV with PCM 16/24/32-bit int or 32-bit float samples
(incl. WAVE_FORMAT_EXTENSIBLE) at any sample rate and channel count —
multi-channel audio is downmixed by averaging and resampled to the tower's
16 kHz mono when the source rate differs.

## Tower-only smoke run (no decoder needed)

`--embd-out` (without `--gen`, `--logits-out`, or `--compare-logits`) and
`--bench` with a media flag return after the tower encode — the decoder
never runs, so the mmproj GGUF itself satisfies the positional
`<model.gguf>` argument. This is the smallest end-to-end run and needs only
the mmproj file:

```sh
zig build inkling -Doptimize=ReleaseFast -- models/mmproj-BF16.gguf \
  --mmproj models/mmproj-BF16.gguf --image photo.png \
  --prompt "<__media__>" --embd-out img_embd.bin
zig build inkling -Doptimize=ReleaseFast -- models/mmproj-BF16.gguf \
  --mmproj models/mmproj-BF16.gguf --audio clip.wav \
  --prompt "<__media__>" --bench 3
```

The image run prints `image: WxH -> RxC patches = N tokens` and writes the
f32 embedding rows (`media embeddings written: … (N x n_embd)`); the audio
run prints `audio: N samples -> M frame tokens`, and `--bench R` reports
best-of-R preprocess/encode times. Full multimodal generation additionally
needs a decoder whose hidden size matches the mmproj embedding width
(`error.MmprojWidthMismatch` otherwise).

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, `-Dcpu`), GPU offload, and the
global thread/BLAS knobs are documented in
[../../docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). This runner
also accepts `--threads N` and `--bench <reps>` directly.

## Parity oracle

`tools/fetch_refs.sh llama.cpp-inkling` pins the reference; the oracle
build recipe (cmake for `refs/llama.cpp-inkling/build-cpu`, incl. the
`llama-completion`/`llama-tokenize` targets and the `tools/llama_logits.cpp`
note) is in the comments of
[`tools/fetch_refs.sh`](../../tools/fetch_refs.sh). `--patch` applies the
two `tools/ref-patches/llama.cpp-inkling-*.patch` files the tower parity
dumps need (dMel width un-hardcoded; mmproj/decoder width-mismatch gate for
tower-only embedding dumps).
