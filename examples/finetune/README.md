# finetune — LoRA fine-tune → merge → re-quantize → serve

LoRA fine-tuning on a real Qwen3 GGUF, on CPU, plus the full loop that turns
the result into a servable model: fine-tune, merge the adapters into dense
weights, quantize the merged model in a second pass, then chat with it or
serve it over HTTP.

Entry point: [`main.zig`](main.zig) (`zig build finetune`).
The merge/quantize passes are `zig build export-gguf`
([`tools/export_gguf.zig`](../../tools/export_gguf.zig)); serving is
`zig build qwen3` or `zig build lmserve`.

## What it does

Trains LoRA adapters on the q and v projections (the classic LoRA-paper
target set) with AdamW while the base model stays frozen — a quantized base
is fine. The built-in SFT dataset has a distinctive style ("Ahoy! … matey."),
so a handful of steps makes the overfit visible: the run prints a BEFORE
(zero-init LoRA) vs AFTER greedy continuation for one held prompt, plus
per-step loss/timing lines. Every run saves a checkpoint directory containing
`adapters.safetensors`, `optimizer.fucina`, and `trainer_state.json`;
`--load` resumes it, including the data-loader position (the sample order
continues instead of restarting at pair 0).

`--data PATH.jsonl` swaps in your own instruction/response pairs
(`src/llm/data.zig`; JSONL schema in
[es_finetune's Custom-data section](../es_finetune/README.md#custom-data---data));
`--verify-grads` replaces the training run with a
quantitative gradient audit through the full production path (zero-structure
at init, per-adapter grad-norm audit, first-order Taylor test, frozen-base
ablation, held-out generalization).

## Getting the weights

Any Qwen3 dense GGUF works (see
[Getting the weights](../../docs/RUNNING-MODELS.md#getting-the-weights) for
all sources). The official `Qwen/Qwen3-0.6B-GGUF` repo ships Q8_0 only; the
full K-quant ladder (Q4_K_S … Q6_K + bf16) is on
[`bartowski/Qwen_Qwen3-0.6B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF)
or [`unsloth/Qwen3-0.6B-GGUF`](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF)
(same pattern per size). bartowski prefixes files with `Qwen_`; every runner
takes a plain path, so rename or adjust as you like.

```sh
mkdir -p models
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_S.gguf --local-dir models

# Merge base for step 2: neither repo ships a plain f16 — download the bf16
# (works directly as the merge base, or transcode an f16 from it — note below)
hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-BF16.gguf --local-dir models
```

The merge step additionally needs a float (f32/f16/bf16) base of the same
model; if your source only ships bf16, transcode an f16 locally — the exact
command is the f16 note in
[Getting the weights](../../docs/RUNNING-MODELS.md#getting-the-weights).

## The loop

### 1. Fine-tune

```sh
# LoRA fine-tune a Qwen3 GGUF on CPU (built-in pirate-style SFT set; ~0.9 s/step
# on 0.6B, M1 Max; loss reaches ~2e-4 by step 30 — TRAINING.md §9)
zig build finetune -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-Q4_K_S.gguf \
  --steps 30 --rank 8 --alpha 16 --save /tmp/qwen3-lora

# Useful extras: --lr F  --seq-max N  --checkpoint-layers (activation checkpointing)
#                --load PATH (resume)  --seed N  --verify-grads (gradient-evidence audit)
#                --accum-steps N (gradient accumulation windows, exact token-weighted; recipe in TRAINING.md §4)
#                --state-dtype f32|bf16 (bf16 optimizer moments; TRAINING.md §3/§8)
#                --data PATH.jsonl  --shuffle  --data-seed N (SFT data via src/llm/data.zig;
#                resume CONTINUES the data order — loader state lives in trainer_state.json)
```

### 2. Merge, then re-quantize

```sh
# Merge the adapters into dense weights (merge needs an f32/f16/bf16 base — a quantized
# base errors; see the f16 note in "Getting the weights". Merge and --dtype are separate
# passes by design: one combined pass would chain-requantize.)
zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf models/Qwen3-0.6B-f16.gguf \
  --out tuned-f16.gguf --adapters /tmp/qwen3-lora --alpha 16

# Quantize the merged model in a second pass (runs in llama.cpp too)
zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf tuned-f16.gguf \
  --out tuned.gguf --dtype q8_0
```

`--adapters` takes the checkpoint directory (or the `adapters.safetensors`
inside it) and requires `--alpha` — the safetensors file stores the adapter
A/B matrices but not alpha, so pass the training-time value.

`export-gguf` also re-emits/transcodes without adapters
(`--dtype f16|bf16|f32|q8_0|q4_k|q5_k|q6_k|verbatim`) and PTQTP-quantizes
tensor-at-a-time with `--ptqtp[=K]` — models far bigger than RAM stream
from the source mmap into `<name>.ptqtp0..K-1` trit-plane tensors that the
family loaders pair-detect (`--ptqtp-include/--ptqtp-exclude` name filters,
`--dry-run` plan preview; [docs/PTQTP.md](../../docs/PTQTP.md)).

### 3. Serve

```sh
# Serve the result (CLI chat, or over HTTP — see "OpenAI-compatible LM server")
zig build qwen3 -Doptimize=ReleaseFast -- tuned.gguf --chat "Who are you?"
zig build lmserve -Doptimize=ReleaseFast -- tuned.gguf --port 8080
```

The HTTP server is documented in
[the lmserve example](../lmserve/README.md).

## Flags

| flag | default | meaning |
| --- | --- | --- |
| `--model PATH` | `models/Qwen3-0.6B-Q4_K_S.gguf` | base GGUF (frozen; may stay quantized) |
| `--steps N` | 30 | optimizer steps |
| `--lr F` | 1e-3 | AdamW learning rate |
| `--rank N` | 8 | LoRA rank |
| `--alpha F` | 16 | LoRA alpha (pass the same value to the merge) |
| `--seq-max N` | 256 | token cap per encoded training pair |
| `--checkpoint-layers` | off | activation checkpointing (TRAINING.md §7) |
| `--accum-steps N` | 1 | gradient-accumulation window, exact token-weighted (§4) |
| `--state-dtype f32\|bf16` | `f32` | optimizer-moment dtype (§3/§8) |
| `--data PATH.jsonl` | built-in set | JSONL SFT dataset (`src/llm/data.zig`) |
| `--shuffle` | off | deterministic per-epoch shuffle |
| `--data-seed N` | `--seed` | shuffle seed |
| `--save DIR` | `/tmp/fucina-qwen3-lora` | checkpoint directory |
| `--save-every N` | 0 (final save only) | periodic checkpoint interval |
| `--load DIR` | — | resume a checkpoint (continues the data order) |
| `--seed N` | 42 | RNG seed |
| `--verify-grads` | off | gradient-evidence audit instead of training |

## Gradient-free variant

`zig build es-finetune` is this example's evolution-strategies twin — same
data/checkpoint plumbing, no backward pass, no optimizer state; see
[`examples/es_finetune`](../es_finetune/README.md). Its lora checkpoint is
the same `adapters.safetensors`, so the merge → quantize → serve loop above
applies unchanged.

## Further reading

- [docs/TRAINING.md](../../docs/TRAINING.md) — the full training guide: §9
  LoRA + LLM fine-tuning, §4 gradient accumulation, §3/§8 optimizer state and
  checkpoint files, §7 activation checkpointing.

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, `-Dcpu`), global thread/BLAS
knobs, GPU offload, MoE expert streaming, and `-Dllguidance` constrained
decoding are documented centrally in
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). The serve steps
(`qwen3`, `lmserve`) accept all of them.
