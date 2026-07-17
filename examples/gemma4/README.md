# Gemma 4 26B-A4B — MoE chat

Runs the Gemma 4 26B-A4B instruction-tuned MoE from a single GGUF. Same
chat/REPL UX as the qwen3 runner, Gemma's `<|turn>` template, SPM tokenizer
read from the GGUF. Build step: `zig build gemma4`; runner source:
[`main.zig`](main.zig).

## Getting the model

Model weights are not part of this repository.

```sh
mkdir -p models
hf download unsloth/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-UD-Q6_K.gguf --local-dir models
```

(`hf` comes from `pip install -U huggingface_hub`; the Q6_K file is 23.2 GB.)

**Gemma license.** Gemma-family weights (Gemma 4, DiffusionGemma) are
distributed under Google's Gemma Terms of Use. The `google/…` originals on
Hugging Face are gated behind accepting those terms; the unsloth GGUF
conversions were not gated at the time of writing, but the terms still apply
to the weights either way.

## Chat and REPL

```sh
# Single-turn chat (sampling defaults come from the GGUF: temp 1.0, top-k 64, top-p 0.95)
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --chat "Why is the sky blue?"

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
```

`--gen N` does raw token generation, `--info` prints the config.

## Zero-copy expert load (`--experts=borrow`)

`--experts=borrow` maps the MoE experts zero-copy instead of x4-packing them.
Q6_K experts otherwise copy+widen ~20 GB on load (slow, doubles memory, can
swap on <48 GB boxes); borrow loads in ~2-3 s at ~half the RSS. Default is
`pack` (peak CPU throughput). Numerically identical (same parity-tested
kernels).

```sh
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --chat "Why is the sky blue?" --experts=borrow
```

## Sampling and constrained decoding

Sampling flags mirror the qwen3 runner (`--temp --top-k --top-p --min-p
--repeat-penalty --repeat-last-n --freq-penalty --presence-penalty --seed
--greedy`) — see [`../qwen3/README.md`](../qwen3/README.md). Constrained
decoding mirrors qwen3 too (`--json-schema/--lark/--regex` on
`--chat`/`--repl`, `-Dllguidance=true` builds); usage guidance is in
[`../../docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md).

## Tokenize, bench, logit parity

```sh
# Encode-only (prints token ids without loading the 22 GB of weights)
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  --prompt "Hello world" --tok-only

# Prefill/decode benchmark + per-block profile; logit parity from raw ids
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  2,651,235 --bench 3 --profile
zig build gemma4 -Doptimize=ReleaseFast -- models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  2,651,235 --logits-out /tmp/g4.bin
```

The raw-ids logits path also takes `--compare-logits <ref.bin>` (compare
against a raw little-endian f32 dump) and `--repeat R` (re-run the forward
pass R times).

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, build on the machine you run on),
GPU offload (`-Dgpu=metal`/`-Dgpu=cuda`), thread/BLAS knobs, and
`-Dllguidance` constrained-decoding guidance are shared across runners — see
[`../../docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md). On
`-Dgpu=metal` builds the batched MoE expert FFN offloads with no flag needed
(see the Metal section there).
