# DiffusionGemma 26B-A4B — block text-diffusion

Runs the DiffusionGemma 26B-A4B MoE. Not autoregressive: it denoises
256-token canvases with the entropy-bound sampler (defaults from the model:
≤48 steps, temperature 0.8→0.4, bound 0.1) and commits blocks
autoregressively. Expect a few seconds per denoising step on the 26B MoE;
simple answers converge in ~7 steps. Build step: `zig build diffusion-gemma`;
runner source: [`../diffusion_gemma.zig`](../diffusion_gemma.zig).

## Getting the model

Model weights are not part of this repository.

```sh
mkdir -p models
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q6_K.gguf --local-dir models
```

(`hf` comes from `pip install -U huggingface_hub`; `…-Q6_K.gguf` is 22.7 GB,
`…-Q4_K_M.gguf` 16.8 GB.)

**Gemma license.** Gemma-family weights (Gemma 4, DiffusionGemma) are
distributed under Google's Gemma Terms of Use. The `google/…` originals on
Hugging Face are gated behind accepting those terms; the unsloth GGUF
conversions were not gated at the time of writing, but the terms still apply
to the weights either way.

## Chat and REPL

On a TTY, chat replies **denoise live inline** — the streaming equivalent for
diffusion: the reply repaints in place where it belongs in the transcript,
with not-yet-accepted tokens faint (they "crystallize" as the sampler
converges) and a dim trailing status line
(`… step 3/48 · accepted 201/256 · H̄ 0.522`); when the block finalizes, the
clean text simply remains, followed by a dim stats trailer. `--no-visual`
disables it, `--visual` forces it when piped, `--visual-interval N` redraws
every Nth step.

```sh
# Chat (Gemma 4 turn template; reply denoises inline on a TTY)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "Why is the sky blue? Answer in two sentences." --max 256 --seed 42

# Multi-turn interactive REPL (context carries across turns; empty line or Ctrl-D quits;
# like llama.cpp -cnv, each turn re-encodes the full history)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --repl --system "Answer tersely."

# Longer multi-block generation (each 256-token canvas is re-encoded into the KV cache)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "Write a 400-word short story about a lighthouse keeper." --max 512 --seed 7
```

## Sampler knobs

```sh
# Tune the entropy-bound sampler (higher bound = more tokens accepted per step = faster/riskier)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "..." --steps 32 --entropy-bound 0.2 --t-max 0.9 --t-min 0.4

# Disable self-conditioning (slightly cheaper, usually worse convergence)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "..." --no-sc
```

Other knobs: `--confidence F` and `--stability N` (the adaptive stop),
`--system "..."`, `--think`.

## Zero-copy expert load (`--experts=borrow`)

`--experts=borrow` maps the MoE experts zero-copy. The Q6_K model otherwise
x4-packs ~20 GB on load (this is the slow/swappy default on memory-tight
boxes); borrow loads it in ~2.5 s at ~half the RSS. Default `pack` favors
peak throughput.

```sh
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --chat "Why is the sky blue? Answer in two sentences." --experts=borrow
```

## Raw generation and info

```sh
# Raw-token block generation (no chat template)
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --gen 256 2,818,7217,7412

# Config/tokenizer info without loading weights
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf --info
```

## Logit-parity harness

```sh
# Logit-parity harness vs llama.cpp PR #24423's llama-diffusion-gemma-eval
# (prompt ids + EXACTLY canvas_length=256 canvas ids; dumps raw f32 [256, vocab])
zig build diffusion-gemma -Doptimize=ReleaseFast -- models/diffusiongemma-26B-A4B-it-Q6_K.gguf \
  --eval 2,651,235 --canvas <256-comma-separated-ids> \
  --logits-out /tmp/dg.bin --compare-logits /tmp/oracle.bin
# add --sc-logits prev.bin to feed a previous step's logits as self-conditioning (temp_inv=1)
```

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, build on the machine you run on),
GPU offload (`-Dgpu=metal`/`-Dgpu=cuda`), and thread/BLAS knobs are shared
across runners — see
[`../../docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md). This runner
additionally accepts `--gpu-f16` on `-Dgpu=metal` builds (dense weights
resident as f16 so the big canvas GEMMs offload; +~4.6 GB — see the Metal
section there).
