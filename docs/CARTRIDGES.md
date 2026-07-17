# Cartridges: trained KV-prefix corpus compression

Fucina's implementation of **Cartridges** (Stanford Hazy Research,
arXiv 2506.06266, "Cartridges: Lightweight and general-purpose long context
representations via self-study"; reference code `HazyResearch/cartridges`).
A cartridge compresses a corpus into the KV cache of a virtual p-token
prefix, trained offline by self-study distillation and served as a reusable
prefix — the paper reports ICL-level quality at ~38× less KV memory and
~26× higher serving throughput.

The method needs three things in ONE system: autograd through the production
transformer (the cartridge is trained by backprop through frozen weights),
KV-cache plumbing that can host a prefix that no tokens produced, and a
serving path that decodes behind it. Fucina has all three, so the whole
flow — training, persistence, serving — runs in-process with no external
services.

## Semantics (pinned to the reference implementation)

- **Parameterization** (`src/llm/cartridge.zig`): per layer, a
  `[p, kv_head, d]` key/value pair in the exact space of KV-cache rows —
  keys post q/k-norm and post-RoPE, rotated at positions `0..p-1` and never
  rotated again. Real tokens sit at RoPE positions `p..`. Row 0 is a frozen
  constant (`frozen_prefix = 1` default — the paper's attention-sink freeze,
  App. A.1: training it destabilizes the run); the rest are leaf variables.
  This is "a simplified version of prefix-tuning": full-dimension rows, no
  MLP reparameterization.
- **Attention**: `concat(sink, trainable, tokens)` along `.seq` feeds the
  stock fused `groupedAttention`, whose end-aligned causal kernel
  (`source_offset = kv_seq − q_seq`) IS the reference block mask: every
  query sees the whole cartridge and is causal over real tokens. The f32-KV
  backward routes gradients through the concat into the trainable rows.
- **Initialization**: the model's OWN K/V rows for the first p corpus
  tokens wrapped as a system message (`Trainer.captureKv` → `initCartridge`),
  the paper's winning strategy (55.3% vs 29.9% random-vector on LongHealth).
  With zero training steps the cartridge is behaviorally identical to a real
  prefill of those tokens — the acceptance gate below.
- **Objective** (`cartridge.distillLoss`): teacher top-k cross-entropy
  `mean(-p_teacher · log q_student)` over sparse `(position, token, logprob)`
  entries — the reference train.py loss, gradient-identical to forward
  KL(teacher ‖ student). Teacher = the same frozen model with the corpus
  chunk really in context; student = the model behind the cartridge; only
  assistant tokens are supervised, read from logits row `position − 1`;
  top-20 entries truncated at 0.99 cumulative mass, tail dropped, NOT
  renormalized. Optimizer: Adam, lr 2e-2, no weight decay, no schedule.
  The trainer computes this through the fused `linearDistillExt` core op:
  the output projection and the sparse targets run as ONE node, only the
  supervised rows are ever projected, and the `[seq, vocab]` logits never
  enter the autograd graph (the composed `cartridge.distillLoss` tail
  remains available — `FUCINA_NO_FUSED_DISTILL=1` — and the two agree to
  f32 roundoff, pinned by a trainer test).
- **Self-study** (paper Sec 4, Algorithm 1 with k = 1;
  `examples/cartridge/main.zig`): sample a uniform random corpus token span and
  one of seven seed-prompt types: the reference five (structuring /
  summarization / question / use-case / creative, the reference meta-prompt
  texts) plus two corpus-generic additions, `mechanism` (why/how reasoning)
  and `verbatim precision` (exact wording/numbers). Seeds carry only the
  FORM of the question, never content: the sampled chunk is the sole
  content selector, which keeps every question answerable from what the
  teacher can see. (Content-bearing seeds — e.g. topics extracted from a
  document up front — break that grounding: whenever the topic misses the
  sampled chunk, bot B answers from parametric memory and distillation
  pumps guesses instead of corpus into the rows.) Bot A (the same model,
  temperature 0.6) writes a chat message about the chunk; bot B (greedy)
  answers with the chunk in context; B's tokens are teacher-forced
  through the frozen model (`evalLogitsRows` — a memory-bounded
  `[rows, vocab]` slice instead of the full logits block) to produce the
  distillation targets.
- **Serving** (`Cartridge.writeToCache`): the full prefix is written into an
  empty `KvCache` (converted to the cache dtype) and the cache advanced to
  p; a normal `forwardStep` decode continues at position p — the exact
  training-time layout. `initFromStateDict` rebuilds a cartridge from
  `saveState` safetensors bytes alone (geometry recovered from the header).

## Where things live

| Piece | Location |
|---|---|
| Cartridge type, distillation loss, targets builder, persistence, serving write | `src/llm/cartridge.zig` (§13.10 in `docs/REFERENCE.md`) |
| Training seams: `ForwardOptions.{cartridge, capture}`, `Trainer.{initCartridge, captureKv, distillLoss, evalLogitsExt, evalLogitsRows}`, (offset, len)-keyed rope tables | `src/llm/qwen3/train.zig` |
| gemma4 training seams (same surface; SWA windows, dual-theta + rope-factor tables, MoE layers, per-layer heterogeneous KV geometry via `Cartridge.initFromRowsVaried`; composed distill tail — soft-capped/quantized heads have no fused route) | `src/llm/gemma/gemma4_train.zig` |
| CLI: `--equiv` gate, self-study training, `--load`/`--ask` serving | `examples/cartridge/main.zig` (`zig build cartridge`) |
| HTTP serving: lmserve `--cartridge` — every conversation preloads the prefix; slot reuse offsets past it (`Conversation.notePrefixRows` / `WarmState.prefix_rows`) | `examples/lmserve/main.zig`, `examples/lmserve/backend.zig`, `src/llm/chat.zig` (`docs/LMSERVER.md`) |
| Mechanism tests + torch 2.12 golden (`tools/gen_cartridge_goldens.py`) | `src/llm/cartridge_tests.zig`, `src/llm/cartridge_golden_tests.zig` |
| qwen3-level gates (equivalence, training smoke, serving parity, roundtrip) | `src/llm/qwen3/train_cartridge_tests.zig` |
| Composition: `composedP` / `validateComposition` / `composedCatK/V` / `writeComposedToCache` / `Cartridge.appendToCache` | `src/llm/cartridge.zig` |
| Composed-forward trainer seams: `ForwardOptions.cartridges`, `Trainer.distillLossExt`, `Trainer.embedLastHidden` | `src/llm/qwen3/train.zig`, `src/llm/gemma/gemma4_train.zig` |
| Fleet: manifest, RAM/disk budget manager, cosine chunk index, mmap artifact retrieval | `src/llm/cartridge_fleet.zig` |
| Fleet CLI: mixed-visibility self-study, index build, retrieval serving, `--equiv` composition gate | `examples/cartridge_fleet/main.zig` (`zig build cartridge-fleet`) |
| Composition + fleet gates | `src/llm/qwen3/train_cartridge_compose_tests.zig`, gemma4 compose arms in `src/llm/gemma/gemma4_train_tests.zig`, `src/llm/cartridge_fleet_tests.zig` |

Use `Trainer(.{ .q = false, .v = false })`: no LoRA adapters, so the
cartridge rows are the only trainable parameters and the base model stays
frozen. Capture, packed, and composed forwards are plain-path only
(`CartridgeCheckpointUnsupported` under `checkpoint_layers`); a SINGLE
cartridge on the no-adapter qwen3 trainer CHECKPOINTS — the rows ride as
block inputs, recompute is pinned bitwise, and both CLIs expose it as
`--checkpoint` (the fleet CLI forces isolated visibility under it).

## Evidence

- **Torch parity** (mechanism): an independent PyTorch implementation of the
  full mechanism (offset RoPE, sink freeze, end-aligned mask, GQA, pos−1
  sparse top-k loss) pins forward logits, loss, and both cartridge gradients
  (`cartridge_golden_tests.zig`, torch 2.12, sha-audited data section).
- **Gradcheck**: finite differences through concat + fused attention into
  the prefix rows (`kv_seq > q_seq`, GQA), and through the distillation loss
  with duplicate positions.
- **Prefill equivalence**: an untrained corpus-init cartridge reproduces the
  real prefill — exactly on the tiny synthetic model, and on
  **Qwen3-0.6B-f16** bitwise (`max |Δlogit| = 0`) at tiled-attention shapes
  (p = 256, suffix 128) with full greedy agreement at every size tried
  (`zig build cartridge -- --corpus README.md --p 256 --equiv`).
- **Serving parity**: decode behind `writeToCache` matches the trainer's
  cartridge eval within f16-KV tolerance on the tiny model.
- **End-to-end behavior** (Qwen3-0.6B-f16, README.md as corpus): a 128-row
  cartridge answers "What is Fucina?" correctly ("a CPU-first
  tensor/autograd runtime and LLM inference engine written in pure Zig…")
  while the bare model hallucinates (Italian pasta) and the ICL reference
  needs 5314 KV rows — 41× the cartridge — for the same answer.

## Demo run-book

The exact reproduction commands — with expected output and measured
wall-clock (~3.5 min for the training command on an M1 Max; self-study
itself is ~6 s/conversation) — live in `examples/cartridge/README.md`
(fleet run-book: `examples/cartridge_fleet/README.md`). The short form:

```bash
# Zero-training acceptance gate on a real GGUF (bitwise at these shapes):
zig build cartridge -Doptimize=ReleaseFast -- \
  --model models/Qwen3-0.6B-f16.gguf --corpus README.md \
  --p 256 --suffix-max 128 --equiv

# Self-study training (fully in-process, 32 conversations ~ 3.3 min) + save:
zig build cartridge -Doptimize=ReleaseFast -- \
  --model models/Qwen3-0.6B-f16.gguf --corpus README.md \
  --p 256 --steps 8 --chunk-min 256 --chunk-max 512 --max-q 64 --max-a 160 \
  --seed 7 --save /tmp/cartridge.safetensors

# Serve it: --ask answers behind the cartridge, bare, and (with a corpus)
# with the real corpus in context — the three-way comparison:
zig build cartridge -Doptimize=ReleaseFast -- \
  --model models/Qwen3-0.6B-f16.gguf --load /tmp/cartridge.safetensors \
  --corpus README.md --ask "What is Fucina, in one sentence?"
```

## Batched self-study (measured)

Three batching layers, measured at the demo recipe (README corpus,
p = 256, accum 4, Qwen3-0.6B-f16):

- **Lockstep batched generation** — the group's bot-A and bot-B decodes
  run as parallel streams through `forwardStepBatch` (finished streams
  retire from the batch). This is the dominant win: the self-study loop is
  decode-bound and batch-N decode amortizes each weight read N ways.
  **17.0 → 10.8 s/conversation (1.57×)** vs the sequential pipeline.
- **One packed teacher pass** — every conversation's top-k targets come
  from a single `evalLogitsRows` over the packed teacher sequences.
- **Packed training step** — one forward/backward for the group
  (`distillLoss(..., packed_segments, ...)`); ≈ neutral vs the flat arm
  at 0.6B (Accelerate's GEMMs are already saturated at single-conversation
  M), kept for the single backward and larger models. `--no-pack` keeps
  the flat-memory per-conversation backward — the packing knob is a
  memory choice, not a speed choice.
- **`--gen-batch N`** decouples generation width from the optimizer group:
  conversations synthesize as ONE lockstep decode group (and one packed
  teacher pass) of N streams, queued and consumed `--accum` at a time, so
  the per-token weight stream amortizes further without growing the packed
  backward. Measured at 1.7B-bf16 on Metal: accum 4 alone 12.3
  s/conversation, `--gen-batch 16` 10.1.

`--spec-b` decodes bot B speculatively — COMPOSED with batching, not
instead of it (single-stream speculation would surrender the batch, a
strictly worse trade): every still-active stream contributes one
[carry ++ draft] span to a single ragged `Model.forwardStepBatchSpans`
weight pass (packed GEMMs; attention per stream against its own cache —
the trainer's packed-segments decomposition, inference-side, gated by a
per-stream equivalence test), drafts come from each stream's lossless
speculation index seeded by its own prompt (the chunk drafts itself),
and greedy verification is exact argmax equality per row with
`truncate` rewinds. At 0.6B the composed form runs at **parity (6.9 vs
6.8 s/conversation)** with byte-identical generations; the net win
scales with draft acceptance: larger models (costlier rows, more
valuable drafts) and quote-heavy corpora tip it positive. Opt-in; the
loop carries a lean per-stream acceptance gate (drafts off below 35%
rolling acceptance over 16 drafted tokens, 32-token re-probe) — the
CostGate idea for the ragged batch.

`--spec-serve` applies the same idea where it pays most, exactly the
cartridge's home turf: serving is single-stream (no batch to trade away)
and the CORPUS itself is the speculation reference, so corpus-grounded
answers — which a trained cartridge produces near-verbatim — draft
themselves. The reference is a property of the ARTIFACT, not of the
serving call: training with `--draft-ref` embeds the corpus token ids in
the saved safetensors (a frozen i64 `draft_reference` entry, 8
bytes/token — the state-dict layer carries i64 alongside f32/f16/bf16
for exactly this kind of frozen metadata), and at serve time the suffix
automaton is built ONCE at cartridge load (`addReference`, ~1 ms per 5k
tokens) before any generation; nothing is constructed per call, and no
`--corpus` (no re-read, no re-tokenize) is needed to serve. Artifacts
without the entry fall back to drafting from `--corpus`; artifacts with
it stay loadable everywhere (`initFromStateDict` recovers it, lmserve
`--cartridge` accepts it). Measured on a full-docs cartridge (p = 1024):
+12–16% tokens/s on long corpus-grounded answers, parity on short ones,
byte-identical output. The acceptance gate is not optional here: without
it a low-acceptance long answer loses ~35% (every rejected draft row
pays attention over the 1024-row prefix plus a vocab-wide logits row).

Reasoning tokens vs speculation (design note): when a serve path emits
template-marked reasoning (`<think>…</think>` are single deterministic
token ids), the three speculation decisions separate. OBSERVE always —
feeding think tokens to the per-request index is lossless and pays for
intra-reasoning self-repetition (the recycling case). GATE per region —
the lean gate's acceptance state should not cross regions: acceptance
collapses inside a think block (compositional prose, not corpus quotes),
and an "off" state carried into the answer region would suppress
drafting exactly where corpus-grounded quotes accept; on marker
crossings the gate counters should swap to a per-region set. DRAFT
everywhere and let the region-scoped gate demote think regions on its
own — hard-disabling inside think forfeits the self-repetition wins.
This belongs to the lmserve per-request speculation package (reasoning
is a per-request toggle there); the cartridge CLI disables thinking by
template, and the reasoning-STYLE prose a small model leaks without
markers is untouchable by template logic anyway (the acceptance gate is
the only defense). Adjacent hardening note: bot B answers are supervised
raw, so a model that emits marked think spans despite the empty-think
template would distill them into the cartridge — stripping marked spans
from B before supervision is cheap insurance, though the reference does
not do it and unmarked reasoning-style prose is beyond any stripper.

## Learning rate vs batch size (measured)

The paper's Adam lr 2e-2 is calibrated to packed batches of 32×2048 tokens
(~65k supervised tokens per optimizer step). At the CLI's demo budgets
(a few conversations ≈ 1k answer tokens per step) that lr measurably
DEGRADES a p = 256 corpus-init cartridge on Qwen3-0.6B-f16 within 25 steps
(think-marker loops, drifted answers), while **lr 2e-3 with `--accum 4`
trains cleanly**: held-out distill loss improves and the trained cartridge
answers "What is Fucina?" word-for-word like the 21×-larger ICL context.
The CLI defaults to 2e-3; scale lr with supervised tokens per step if you
raise the batch toward the paper's regime.

## What a cartridge learns (measured)

Two budgets bound answer quality, measured on this repo's own
documentation as the corpus:

- **Coverage** (conversations × mean chunk length / corpus tokens) bounds
  verbatim recall. At ~2.4× coverage of a 330k-token corpus, a 0.6B
  cartridge recalls opening facts and isolated numbers but misses most of
  the long tail; at ~20× coverage of a single ~5k-token document, it
  quotes the document's actual sentences. Per-conversation cost is
  corpus-size independent, so coverage is bought purely with conversation
  count.
- **The teacher's ICL ceiling** bounds mechanism-level correctness:
  distillation cannot exceed what the teacher answers with the chunk in
  its context. Qwen3-0.6B answers why/how mechanism probes incorrectly
  even with the whole source document in context, and its cartridges
  inherit that; Qwen3-1.7B answers them essentially correctly, and its
  single-document cartridge (256 conversations, p = 512) reproduces the
  correct explanations. Teacher scale — not seed engineering — is the
  lever for knowledge depth, consistent with the paper's Fig. 5.

## Deliberate divergences from the reference

- Teacher logprobs come from a dedicated teacher-forced eval pass
  (`evalLogitsRows`, trainer f32 path) instead of being recorded by the
  generation server — same distribution (B is generated greedily from the
  same context), better numerics, and no server dependency.
- Packing groups the accumulation window's conversations rather than
  cutting fixed 2048-token rows: `ForwardOptions.packed_segments` runs the
  group as contiguous segments of ONE forward (GEMMs batch over the packed
  rows; RoPE restarts per segment; attention runs per segment over
  zero-copy narrows, so the reference's block mask holds bit-for-bit and
  gradients flow through the existing fused backward — packed vs
  sequential gradient equality is a trainer test). `--no-pack` keeps the
  flat-memory per-conversation backward; both arms train identically.
- Multi-document corpora (`--corpus` repeats; directories contribute their
  top-level `.md` files sorted) concatenate into one provenance-tagged
  stream: per-file `# Document: <path>` headers plus the reference-style
  one-line chunk description naming the source file. Per-conversation
  training cost is independent of corpus size — only chunk coverage scales
  with the conversation budget.
- The structuring seed prompt pins `data_format = JSON` rather than sampling
  among six formats; `prob_thinking` is effectively 0 (thinking disabled on
  both bots).

## Acceleration (measured, M1 Max)

- **Metal** (`-Dgpu=metal`) pays on the BATCHED pipeline: batched
  generation turns decode into wider passes the GPU wins, and
  prefill-shaped GEMMs (f16 and bf16 weights) offload. Measured
  end-to-end self-study: **4.2 vs 5.1 s/conversation at 0.6B-f16
  (~1.2×)**, and **9.7 vs 20.7 s/conversation at 1.7B-bf16 with
  `--gen-batch 16` (~2.1×)**. The SEQUENTIAL pipeline is
  decode-dominated and runs ~2.3× slower on Metal at 0.6B — use the
  batched (default `--pack`) pipeline on GPU builds. Offload floors and
  knobs live in `docs/GPU-OFFLOAD.md`.
- **CPU-only builds**: `FUCINA_CPU_F32_SHADOW=1` routes prefill-shaped
  16-bit-weight GEMMs through BLAS over a widen-once f32 shadow —
  **28.5 → 12.7 s/conversation (2.2×) at 1.7B-bf16** with identical
  per-step losses (+4 bytes/weight resident, static weights only; see
  `docs/GPU-OFFLOAD.md`).

## Training memory: `--checkpoint` (measured)

Keep-vs-recompute is a user flag on both cartridge CLIs (default:
keep activations). With `--checkpoint`, the qwen3 no-adapter
single-cartridge forward recomputes each layer in the backward — the
cartridge rows ride as checkpoint-block inputs, and the recompute is
pinned BITWISE (loss and row gradients) by a trainer test. Eval and
capture entries always run plain (recompute buys nothing without a
backward). Measured on Qwen3-1.7B-BF16 (p = 512, 160 supervised tokens):
peak training RSS **15.7 → 7.9 GB** with a byte-identical trained
artifact, and the step is slightly faster (43.0 → 32.5 s — reduced
allocator/page pressure outweighs the ~2x layer recompute at this
scale). The fleet CLI forces isolated visibility under the flag; composed
and gemma4 recompute are follow-ups.

## gemma4 (measured)

The gemma4 trainer carries the same cartridge surface as qwen3
(`initCartridge` / `captureKv` / `evalLogitsExt` / `evalLogitsRows` /
`distillLoss`), covering gemma-4's architecture: local-SWA layers (a
served cartridge is visible to a SWA query exactly as a real prefix —
within the window), dual-theta rope with per-frequency factors on global
layers, MoE FFNs (distillation gradients flow through router + experts
into the prefix rows), and PER-LAYER heterogeneous KV geometry
(gemma-4 26B mixes 8-head/256-dim SWA layers with 2-head/512-dim
globals; `Cartridge.initFromRowsVaried` + per-layer state-dict recovery
carry it end to end). Mechanism gates are exact (1e-5) on tiny models
for all four arms: dense, SWA-cuts-the-prefix, MoE, and rope-factors
with offset positions. Cross-layer shared-KV models are rejected
(`CartridgeGeometry`); packed segments are not routed on gemma4.

On real quantized-MoE weights, logit-level equivalence has an intrinsic
bound that has nothing to do with cartridges: the stack is NOT
shape-invariant across GEMM kernel classes (measured on
gemma-4-26B-A4B Q6_K, no cartridge anywhere: the same 32 rows forwarded
alone vs inside a 96-row batch differ by up to ~12 logits — near-tie
experts flip). The CLI's gemma `--equiv` arm therefore measures the
model's own shape-sensitivity envelope first and judges the cartridge
against it (the cartridge student necessarily runs suffix-shaped GEMMs).

Self-study training runs through the same CLI engine (the chat template,
tokenizer, per-model caches, and packing capability are per-architecture
values; gemma uses the `<|turn>` template with the thought channel primed
off, batched generation via `Model.forwardStepBatchSpans`, `--spec-b`
included, and the flat-memory per-conversation backward — no packed
forward). Measured on gemma-4-26B-A4B Q6_K (native CPU, 64 GB): ~89
s/conversation at smoke budgets with held-out distill loss improving from
the first step; peak RSS ~46 GB — the backward through the frozen
quantized experts carries a large transient (a streamed dX for quantized
weights is the recorded follow-up), so training wants headroom. Metal
builds duplicate the expert blocks into device allocations at load and do
NOT fit 26B training on 64 GB — train quantized-MoE models on CPU builds
(`borrow_experts` keeps the weights zero-copy); serving is unaffected.
Serving also runs here (`--load`/`--ask`) and through lmserve
`--cartridge`.

## Cartridges at Scale: per-document fleets (arXiv 2606.04557)

Fucina's implementation of the follow-up paper ("Cartridges at Scale:
Training Modular KV Caches over Large Document Collections"): instead of
one monolithic cartridge per corpus, every document gets its OWN
cartridge; the pieces are trained jointly so they still work when several
are loaded together, rotate between RAM and disk under a budget, and are
selected per query by an in-process retriever. The paper reports
per-document fleets beating monoliths by 10–31 points at matched token
budgets, with mixed-visibility training the ingredient that stops
composed serving from collapsing.

### Composition

Independently trained cartridges compose by CONCATENATION: part 0's rows,
then part 1's, …, real tokens at RoPE positions `composedP(parts)..`.
Every part keeps the rotations it was trained at — post-RoPE keys are
frozen vectors, so overlapping nominal positions across parts are
harmless (attention only ever takes dot products against them).

- `cartridge.zig`: `composedP` / `validateComposition` / `composedCatK/V`
  (one concat per layer over every part's sink + trainable rows plus the
  token rows) / `writeComposedToCache` (serve: parts in order into an
  empty cache; `Cartridge.appendToCache` is the primitive). Serving-side
  composition is cache-level and therefore model-agnostic — gemma4
  cartridges compose at serve time today even though fleet TRAINING is
  qwen3-typed (recorded deferral below).
- Trainer seams (qwen3 AND gemma4): `ForwardOptions.cartridges` (mutually
  exclusive with `cartridge`; plain-path only on qwen3) +
  `Trainer.distillLossExt`. Because the composed prefix is ONE concat node
  per layer, the existing fused attention backward routes gradients into
  EVERY part's trainable rows — joint training needs no new autograd
  machinery. On gemma4 the SWA layers see the composed prefix within their
  window exactly like a single prefix of the same length, and
  heterogeneous per-layer KV geometry composes (parts must agree
  layer-for-layer).
- **Evidence**: a single-part composition is BITWISE the single-cartridge
  forward; a two-part composition whose parts come from one capture split
  at p (part B holds rows at positions p..2p−1) reproduces the real
  prefill of the concatenated prefix — exact on the tiny models for BOTH
  families (`train_cartridge_compose_tests.zig`; gemma4 arms in
  `gemma4_train_tests.zig` include an SWA window that cuts through the
  composed prefix and composed serve parity: gradients reach every part,
  sinks stay frozen, packed-vs-flat parity, geometry/checkpoint
  rejections) and **bitwise on Qwen3-0.6B-f16** (`zig build cartridge-fleet
  -- --docs README.md --equiv --p 256 --suffix-max 128`: max |Δlogit| = 0,
  greedy 128/128 over 512 composed rows). On gemma-4-26B-A4B Q6_K the
  composed `--equiv` behaves exactly like the base method's gemma gate:
  greedy flips (28/32) sit INSIDE the model's own shape-sensitivity
  envelope (composition max |Δlogit| 10.9 vs the model's own 11.8 with no
  cartridge anywhere — quantized-MoE stacks are not GEMM-shape-invariant,
  docs/CARTRIDGES.md "gemma4").

### The fleet (`src/llm/cartridge_fleet.zig`)

- **Manifest** (`fleet.json`): per-document cartridge/optimizer file
  names, token counts, and optimizer-step counters — the budget manager's
  eviction currency. Cartridges persist as the base method's safetensors
  state dicts (one per document), optimizer moments as FZT1 snapshots.
- **Budget manager** (`Fleet`): at most `budget` cartridges RESIDENT
  (rows + Adam moments live, each document with its own AdamW); every
  `every` optimizer rounds, `ceil(evict_fraction · residents)`
  MOST-trained residents rotate to disk and the LEAST-trained absentees
  rotate in (ties → lower doc id) — the paper's uniform-coverage policy
  (R = 10, φ = 0.5 defaults, scaled to CLI budgets). Entrants ramp under
  a per-cartridge lr warm-up. **An evict/reload cycle continues training
  bit-identically to never leaving memory** (rows AND moments roundtrip;
  pinned by a fleet test), and all artifact writes are atomic. Artifact
  retrieval is mmap-based (`mmapFile`): reloads stream mapped pages
  straight into fresh tensors with no whole-file heap copy.
- **Mixed-visibility joint self-study** (`examples/cartridge_fleet/main.zig`,
  `zig build cartridge-fleet`; qwen3 AND gemma4 GGUFs — gemma runs the
  flat-memory per-conversation backward and per-conversation teacher
  passes, no packed forward, like the base CLI; NOTE the 26B-scale memory
  reality below): each round samples a target document from
  the residents proportional to its token length; with probability
  `--p-iso` (default 0.75) the target's cartridge is the only prefix,
  otherwise 1..`--distract-max` distractor cartridges from other
  residents co-load in shuffled order. Conversations synthesize with the
  base CLI's engine (bot A asks about a chunk of the TARGET document,
  bot B answers with the chunk in context, one packed teacher pass), the
  packed student forward runs behind the composed prefix, and every
  co-loaded cartridge's optimizer steps (each under its own warm-up lr) —
  distractors learning to stay quiet on other documents' questions is
  exactly the paper's fix for composition collapse.

### Cartridge-RAG selection (no external retrieval stack)

`EmbedIndex` (`index.safetensors`): each document is chunked at
`--embed-chunk` tokens and embedded THROUGH THE SERVING MODEL ITSELF —
the chunk text is suffixed with a one-line "In one word, the main topic
of the text above is:" instruction and the final-norm LAST hidden state
is the embedding. Rows are L2-normalized and then CENTERED at `finalize`
(subtract the chunk centroid, re-normalize — the all-but-the-top
correction; the centroid persists in the index and centers queries too).
Selection is a hand-rolled cosine top-k over chunks (a dot product per
row), deduplicated to documents in best-chunk order; the selected
cartridges load from disk (mmap), compose, and the answer decodes behind
`writeComposedToCache`. `--oracle NAME` bypasses retrieval.

Embedding and teacher-pass forwards run SCOPELESS (`embedLastHidden`,
`evalLogitsRows`): with no exec scope open, the layer bodies'
defer-deinits free every intermediate eagerly, so the peak is the
layer-local working set instead of the whole retained forward — an
ownership-only property, the arithmetic (and therefore the embedding
recipe) is unchanged.

The recipe — topic-instruction suffix + final-norm last hidden state +
centroid centering — is THE CONTRACT: index chunks and serve-time queries
must embed identically (the paper offloads selection to an external dense
retriever and calls a learned cartridge retriever future work; the
in-process recipe keeps the no-dependency contract). Two operational
rules, both measured at 26B: keep `--embed-chunk` at or below 256 —
512-token chunks mis-rank because broad chunks of a document that merely
MENTIONS a topic out-score the topic's own document (0.647 vs 0.579,
against 0.699 vs 0.579 correctly ordered at 256) — and prefer building
the index on the platform that serves it (cross-platform index/query
mixing shifts raw cosines by up to ~0.06; rankings hold, margins shrink).

### Deliberate divergences from the paper

- **RAM/disk instead of GPU/NVMe**: Fucina is CPU-first, so "GPU budget"
  becomes resident-memory budget; moments go to disk files rather than
  NVMe-offloaded optimizer state. Same policy, same counters.
- **Visibility is sampled per optimizer GROUP, not per conversation**:
  the packed forward shares one prefix across its segments, so the
  accumulation group shares one visibility draw (isolation still hits at
  the same rate; distractor sets still vary across groups). Per-segment
  part sets inside one packed row would need per-segment prefixes — a
  recorded follow-up, `--no-pack` gives the flat arm today.
- **Both bots are the target model** (as in the base CLI); the paper
  decouples question generation onto a large external model (GPT-OSS
  120B). Proportional-to-length sampling applies over RESIDENT documents
  (the paper samples over the whole collection when building its QA set
  up front).
- **Steps count for every co-loaded cartridge** (they all received an
  optimizer step), so distractor duty advances a document's rotation
  priority exactly like target duty.

### gemma4 fleets (measured, gemma-4-26B-A4B Q6_K)

Everything runs end to end on real weights. Inference-side, on a 64 GB
M1 Max: the composed `--equiv` gate (greedy flips inside the model's own
shape-sensitivity envelope, 10.9 vs 11.8 — the base method's
quantized-MoE verdict, mechanism pinned exact by the tiny-model gates),
the per-document init pass (~4 s/doc at p = 64), the retrieval index
(41×256-token chunks in 366 s, 8.9 s/chunk, peak RSS 28.8 GB),
retrieval/oracle serving (~9 tok/s decode behind a composed prefix), and
lmserve `--fleet`.

TRAINING runs at ~210 s/conversation isolated and ~220 s co-loaded
(every co-loaded part's optimizer steps), with held-out distill loss
improving from the first rounds (0.9201 → 0.8839 over a 3-round smoke).
It is memory-gated: the backward transient through the frozen quantized
experts peaks at **58–118 GB per conversation** even at smoke budgets
(≤48 supervised tokens), so budget **≥128 GB of RAM** for 26B fleet
training (the streamed quantized-dX and gemma-recompute follow-ups are
the levers that shrink this). Fleet artifacts are platform-portable:
x86-trained fleets serve on ARM unchanged. qwen3-scale fleets train
comfortably on a laptop.

### Fleet demo (measured, M1 Max, Qwen3-0.6B-f16)

4 docs (README + TERNARY + PTQTP + SPECULATIVE, ~21.6k tokens), p = 256,
budget 3, rotate every 6, 24 rounds × accum 4 (96 conversations, 12.3k
supervised tokens): init pass ~0.5 s/doc, ~5.5 s/conversation of
self-study, retrieval index 88 chunks in ~44 s (one forward per chunk).
Rounds mix isolated and co-loaded ×2/×3 visibility, and rotation delivers
uniform coverage — final per-document step counts 8/8/8/9 with only 3 of
4 cartridges ever resident. Serving: one 256-row
cartridge answers in ~0.9 s, a composed pair (512 rows) in ~2.5–3.3 s at
~40 tok/s; retrieval picks the home document ("What is Fucina" → README
→ the canonical one-sentence answer; the ternary question → TERNARY.md
→ ternary-packing content) while the bare model hallucinates ("Italian
dessert", base-3 "quantum computing"). At these smoke budgets held-out
distill loss is flat (0.1202 → 0.1199) — coverage, not mechanism, is the
limit, exactly as with the base method's demo budgets.

### HTTP serving (lmserve `--fleet`)

`zig build lmserve -- <model.gguf> --fleet DIR` serves a fleet behind the
OpenAI-compatible server (docs/LMSERVER.md): per request, the user
messages embed through the model (`Trainer.embedLastHidden` + the
`cartridge_fleet.embed_suffix` contract), the cosine index picks
`--rag-docs` documents, and the selected cartridges compose as the
conversation's prefix. Slot reuse is conversation-STICKY: conversation
identity is the FIRST user message, a continuation keeps the selection
its conversation started with (per-turn re-retrieval is unstable on
runner-up documents and forfeits all KV reuse; token-LCP alone cannot
carry identity — the constant template preamble of unrelated short
prompts passes the similarity gate), and retrieval runs only for
conversations no slot remembers. Follow-up turns report `cached_tokens`
through the composed prefix; interleaved conversations keep distinct
selections. qwen3 and gemma4 backends (gemma4 MoE GGUFs need
`--experts=borrow` — the query embedder forwards through raw expert
blocks); excludes `--kv-cache-dir` (sidecars don't record selections).

`--rag-adaptive` adds in-conversation re-selection with a DECISIVE-margin
rule: a follow-up switches knowledge base only when a document outside
the current selection beats every current document's best chunk by
`--rag-margin` under the contextual (all-user-messages) query; the switch
destroys the conversation's slot and rebuilds the composed prefix. The
rule is deliberately relative and context-anchored — absolute score
floors and last-message-alone probes cannot separate phatic turns from
topical pivots on this retriever — so switches fire on clear cross-domain
shifts and not on noise; on same-register corpora they are rare, and a
NEW conversation (fresh first message) is the reliable hard-pivot. A
stronger retriever, not a different threshold, moves this ceiling (the
learned-retriever follow-up below).

### Follow-ups (not landed)

- **Per-conversation visibility inside packed rows** (per-segment part
  sets), restoring the paper's exact per-example sampling under packing.
- **Learned cartridge retriever**: the paper's own future-work item;
  today's index embeds text chunks, not the trained cartridge tensors.
- **Fleet disk tier for lmserve**: selection-aware KV sidecars (record the
  doc ids next to the prefix rows) would re-enable `--kv-cache-dir` under
  `--fleet`.

## Follow-ups (not landed)

- **GPU training on CUDA**: unmeasured for this workload; the rig's
  larger prefill/decode wins warrant a run, and bf16 weight GEMMs need
  the cuBLAS `CUDA_R_16BF` arm (they currently take the CPU kernels on
  CUDA builds).
- **lmserve per-request speculation** consuming the embedded draft
  reference (a shared immutable automaton at startup; conflicts with slot
  reuse today), plus a sampled batched verify in core.
- **Checkpointing for composed and gemma4 forwards**: composed prefixes
  need a runtime-arity (slice-based) checkpoint-input API in core, and
  the gemma4 trainer has no recompute plumbing — both are required before
  26B gemma fleet training fits small-RAM boxes (the retained forward
  graph is the dominant memory term there; the per-node dequantized
  weight transient in `ConstRhsEinsumBackward` is sub-GB at any instant).
- **Streamed quantized-dX**: the backward re-dequantizes every touched
  frozen quantized weight per step (~17 GB of dequantization traffic per
  26B conversation); a fused quantized-dX kernel — ideally the CUDA-side
  variant — removes that bandwidth wholesale. This is the TIME lever;
  checkpointing is the MEMORY lever.
