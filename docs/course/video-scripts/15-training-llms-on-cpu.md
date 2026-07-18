# Video 15 — Training LLMs on your CPU (3:00)

*Series: Forging Deep Learning in Zig · Source: ../15-training-llms-on-cpu.md*

## Logline
Close the loop most tutorials leave open: LoRA fine-tune Qwen3-0.6B on a
laptop CPU — 1.15M trainable parameters beside a frozen base — then merge,
quantize, and watch the exported GGUF answer in llama.cpp. Plus the
existence proof that CPU training goes past fine-tuning: karpathy's
nanochat, ported whole and trained from scratch.

## Takeaways
1. **Quantized weights are constants** — no `GradState`, by type. LoRA
   trains two thin factors beside the frozen weight; at rank 8 on q+v
   that is ≈1.15M parameters, about 0.19% of Qwen3-0.6B, and gradients
   flow to the f32 activations only.
2. **The loop closes in four commands** — fine-tune → merge → quantize →
   serve. Merge and quantize are separate passes *by design*, so the
   exported file has exactly one quantization in its history — and it
   answers in llama.cpp, which has never heard of any of this.
3. **Honest scope** — CPU training is for fine-tunes, small from-scratch
   models (nanochat: BPE training, pretraining, SFT, chat — verified
   against the Python reference), and determinism-first research loops.
   It is not pretraining at modern scale.

## Script

### [0:00–0:18] The file that leaves home
**VO:** Most fine-tuning tutorials end the same way: a file only their own
framework can read. This one ends in llama.cpp. Three minutes: teach a
quantized Qwen3 a new style, on a laptop CPU, then export a standard GGUF
that any GGUF consumer can serve.
**Visual:** Cold-open terminal recording: `llama-cli -m
/tmp/qwen3-tuned-q4_k.gguf`, the prompt "What is the capital of France?"
typed interactively, and the model's pirate-styled reply appearing (real
capture from the asset-list run — do not script the model's words).
**Overlay:** `trained in Zig on a CPU → answering in llama.cpp`

### [0:18–0:55] The wall, and LoRA around it
**VO:** Why not full fine-tuning? Qwen3-0.6B is 595 million parameters —
AdamW bookkeeping alone runs to gigabytes. And the file you actually have
is quantized: in Fucina, quantized weights are constants. No gradient
state — a type-level fact, not a convention. So LoRA goes around the wall.
Freeze the weight; train two thin factors beside it. B starts at zero, so
at step zero the adapted model is the base model, exactly. Rank 8 on the q
and v projections: 1.15 million trainable parameters. About 0.19 percent
of the model — and optimizer state in megabytes.
**Visual:** Code shot: the LoRA equation block at
`docs/course/15-training-llms-on-cpu.md:30-34` (`y = W·x + (alpha/r) ·
B·(A·x)`, A trainable random, B trainable ZERO init); then the counting
block at `docs/course/15-training-llms-on-cpu.md:115-117` ending in
`1_146_880 ≈ 1.15M`, with the total highlighted.
**Overlay:** `quantized weights carry no GradState — constants, by type` →
`1.15M trainable ≈ 0.19% of 595M`

### [0:55–1:26] Thirty steps of pirate
**VO:** The demo dataset is five pairs; every answer opens with Ahoy.
Gradients flow to the f32 activations only — backward multiplies by the
frozen weight, it never updates it — so the base stays in its Q4_K_S blocks
while a few megabytes of adapter train. Thirty steps later, loss falls
from 5.77 to 2e-4. That is memorization, on purpose. The point is the
whole pipeline closing in under a minute of laptop time — and the
after-generation opens with Ahoy.
**Visual:** Code shot: the five-pair pirate dataset at
`examples/finetune/main.zig:38-44`; then terminal recording:
`zig build finetune -Doptimize=ReleaseFast -- --steps 30` (default model
`models/Qwen3-0.6B-Q4_K_S.gguf`), capturing the BEFORE generation
(pirate-free), the loss trace, and the AFTER generation opening with
"Ahoy!".
**Overlay:** `loss 5.77 → 2e-4 in 30 steps · ~932 ms/step — M1 Max
snapshot (docs/TRAINING.md §9)` · `2e-4 on five pairs = memorization, by
design`

### [1:26–2:12] Four commands and the loop closes
**VO:** Now the loop closes: four commands. One — fine-tune over the dense
f16 base and save the adapters. Two — merge them into the weights; this is
the only place B times A is ever materialized. Three — quantize the merged
file to q4_k. Four — serve it. In Fucina, or in llama.cpp, which has never
heard of any of this. Merge and quantize are separate passes by design: a
quantized base refuses to merge, because every extra trip through a
quantizer compounds error. The exported file carries exactly one
quantization in its history. And the repo's claim is tested: the merged
model answers in the tuned style under llama-cli, and loads in llama-bench
as Q4_K Small.
**Visual:** Terminal recording of the four-command script quoted verbatim
from `docs/TRAINING.md:588-603` (steps numbered 1–4 as in the doc's own
comments): `zig build finetune … --model models/Qwen3-0.6B-f16.gguf
--steps 30 --save /tmp/qwen3-lora`, then `zig build export-gguf …
--adapters /tmp/qwen3-lora --alpha 16 --out /tmp/qwen3-tuned-f16.gguf`,
then `zig build export-gguf … --dtype q4_k --out
/tmp/qwen3-tuned-q4_k.gguf`, then `zig build qwen3 …
/tmp/qwen3-tuned-q4_k.gguf --chat "What is the capital of France?"` and
`llama-cli -m /tmp/qwen3-tuned-q4_k.gguf` (the cold-open's shot, now in
context). Time-compress the runs; keep each command line readable.
Optional cut while step 2 runs: the merge at `src/lora.zig:265-274`
(`w_data += scale * (B·A)`).
**Overlay:** `merge + quantize: separate passes BY DESIGN — one
quantization in the file's history` · `"answers … under llama-cli … loads
… in llama-bench as 'Q4_K - Small'" — docs/TRAINING.md §9`

### [2:12–2:37] nanochat: from scratch, whole
**VO:** And if fine-tuning feels like borrowing someone else's weights,
the tree holds an existence proof: karpathy's nanochat, ported whole.
Tokenizer training, GPT pretraining, supervised fine-tuning, bits-per-byte
eval, and chat with a calculator tool — from scratch, on CPU, verified
against the Python reference: token-exact tokenizer, byte-identical
loaders, per-layer forward parity. It's a small GPT — six layers, 384
wide — and nobody pretends otherwise.
**Visual:** Code shot: the five-subcommand table at
`examples/nanochat/README.md:20-28` (tok-train / base-train / sft /
eval-bpb / chat); then the honest CPU-demo config `d6` at
`examples/nanochat/model.zig:56-65` with `.n_layer = 6` and
`.n_embd = 384` highlighted.
**Overlay:** `parity target: nanochat @ 92d63d4, on CPU in fp32` ·
`example-local — no src/ changes`

### [2:37–3:00] What it's for — and next
**VO:** That is the honest shape of CPU training: fine-tunes of real
models, small models on your own data, determinism-first research loops.
It is not pretraining at modern scale, and the docs never claim it is.
After pretraining: gradients, evolution, trit-planes, or a trained
context. Formats, not frameworks, are the interface. Next time: the
craft — how a library like this earns your trust.
**Visual:** The three-roads table from
`docs/course/15-training-llms-on-cpu.md:646-650` (LoRA+backprop / ES /
from-scratch, one row each); then the post-training menu table at
`docs/course/15-training-llms-on-cpu.md:654-659` (LoRA / ES / PTQTP /
Cartridges, one row each); end card with "Full chapter:
`docs/course/15-training-llms-on-cpu.md`".
**Overlay:** `four roads after pretraining — train the weights,
re-express them, or train the context` · `CPU-first, not CPU-only` ·
`full chapter in docs/course/` · `Next: 16 — The craft`

## Asset list
- **Model downloads** (weights are NOT in the repo; sources per
  `docs/RUNNING-MODELS.md:31,47-51`):
  - `models/Qwen3-0.6B-Q4_K_S.gguf` — from
    `bartowski/Qwen_Qwen3-0.6B-GGUF` (file ships as
    `Qwen_Qwen3-0.6B-Q4_K_S.gguf`; rename) or `unsloth/Qwen3-0.6B-GGUF`
    on Hugging Face. Needed for the 0:55 demo run.
  - `models/Qwen3-0.6B-f16.gguf` — same sources; if only bf16 is
    available, transcode locally first:
    `zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf
    <src>.gguf --out models/Qwen3-0.6B-f16.gguf --dtype f16`
    (`docs/RUNNING-MODELS.md:49-51`). Needed for the 1:26 loop.
- **External tool**: a llama.cpp build providing `llama-cli` (and
  optionally `llama-bench`) for steps at 0:00 and 1:26.
- **Terminal recordings** (all `-Doptimize=ReleaseFast`):
  1. `zig build finetune -- --steps 30` — BEFORE/AFTER generations + loss
     trace (0:55 segment).
  2. The four-command loop exactly as printed in `docs/TRAINING.md:588-603`
     (1:26 segment) — note `--alpha 16` must repeat the training-time
     alpha; the adapter checkpoint does not store it.
  3. `llama-cli -m /tmp/qwen3-tuned-q4_k.gguf`, prompt "What is the
     capital of France?" typed interactively — capture the actual reply;
     this recording doubles as the 0:00 cold open.
- **Repo code shots**: `examples/finetune/main.zig:38-44` (pirate dataset);
  `src/lora.zig:265-274` (merge, optional); `examples/nanochat/README.md:20-28`
  (subcommand table); `examples/nanochat/model.zig:56-65` (d6 config).
- **Chapter shots**: `docs/course/15-training-llms-on-cpu.md:30-34` (LoRA
  equation), `:115-117` (parameter count), `:646-650` (three-roads table),
  `:654-659` (post-training menu table).
- **Diagrams**: none required — the equation and table shots carry the
  structure.

## Production notes
- **Tone**: matter-of-fact wonder. The emotional beat is the cold open
  paying off at 2:05 — the same llama-cli shot, now earned. Let the
  "Ahoy!" AFTER generation land without a joke on top of it.
- **VO count and pacing**: 449 spoken words total — in the 380–450
  contract range. The standalone " — " dashes are pause marks, not words
  (a plain `wc -w` reads 463; do not "fix" that as an overrun). The
  closer paces ~162 wpm; if it must breathe, the one sanctioned VO trim
  is dropping "After pretraining: gradients, evolution, trit-planes, or
  a trained context." — the menu-table shot and its overlay carry the
  same content.
- **Caveats that MUST stay attached**:
  - `~932 ms/step` and `5.77 → 2e-4 in 30 steps` are an **M1 Max
    snapshot** from `docs/TRAINING.md §9` — the overlay keeps that
    framing; they are not promises.
  - "2e-4 is memorization, on purpose" stays in VO — five pairs overfit
    by design; the demo proves the pipeline closes, not model quality.
  - The llama-bench quote is verbatim from `docs/TRAINING.md §9`
    ("Q4_K - Small") — quote exactly or drop.
  - nanochat's parity target keeps its precision: the Python reference
    at `92d63d4`, **on CPU in fp32** — and d6 is a *small* GPT; no scale
    inflation.
  - No state-of-the-art or production-readiness claims anywhere.
- **Recording gotchas**: the fine-tune trains on "What is the capital of
  France?", so the styled reply is expected — but capture the model's
  actual words, never a mock-up. The four-command loop fine-tunes over the
  **f16** base (adapters must match the weights they merge into); the
  0:55 demo numbers come from the default **Q4_K_S** run — don't mix the
  two runs' numbers.
- **If the cut runs long, trim in this order**: (1) the `src/lora.zig`
  merge cut at 1:26; (2) the d6 config shot at 2:12 (keep the subcommand
  table); (3) the parameter-count chapter shot at 0:18 (its overlay
  carries the figure); (4) the dataset code shot at 0:55 (keep the
  BEFORE/AFTER terminal).
- **Do not change**: the four commands' order and flags (verbatim from
  `docs/TRAINING.md §9`, including `--alpha 16`); the
  merge-then-quantize rationale ("every extra trip through a quantizer
  compounds error" — one quantization in the file's history); the
  "not pretraining at modern scale" sentence; "Formats, not frameworks,
  are the interface."
- **Line ranges**: every file/line citation in this script was verified
  against the working tree on 2026-07-17; re-verify at record time if the
  repo has moved since.
- The chapter has much more (the SFT data pipeline and its BPE boundary
  trap, gradient verification from three angles, the ES twin,
  checkpoint/resume contracts); the end card sends viewers there.
