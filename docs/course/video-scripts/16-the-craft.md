# Video 16 — The craft (3:00)

*Series: Forging Deep Learning in Zig · Source: ../16-the-craft.md*

## Logline

How a one-person library earns trust: a verification religion where every
claim of correctness answers to an independent oracle — token-ID-exact
tokenizers, byte-exact encoders, golden optimizers, finite-difference
gradcheck, a scalar backend as executable specification, docs that run in
CI — and a benchmark record honest enough to keep a 0.36× defeat on the
scoreboard, investigate it, and record the residual. Every discipline shown
is transferable to the viewer's own projects.

## Takeaways

1. Build the oracle before the feature: an independent source of truth,
   compared mechanically, exiting nonzero — then freeze the gates into a
   ratchet ("an optimization that flips a single token is reverted, not
   tolerance-adjusted").
2. Enforce what you can by machine (an SCC check over the import graph, docs
   snippets executed in CI) and label what you can't as review-only, in
   writing.
3. Losses recorded as plainly as wins — dated, conditioned, with hypothesis,
   control, and residual — are what make the wins believable. That is a
   decision, not an infrastructure.

## Script

### [0:00–0:24] The quiet failures

**VO:** A tensor library fails quietly. A wrong sign trains slightly worse. A
reordered addition flips a token three layers downstream. A missed SIMD arm
costs most of your throughput — while every test stays green.
Fucina's defense starts with one sentence from its contributing guide: the
library regresses in exactly two ways. It becomes wrong, or it becomes slow.

**Visual:** Dark screen; the three failure modes type on one at a time as the
VO names them ("wrong sign · trains slightly worse", "reordered addition ·
flips a token", "missed SIMD arm · throughput gone, tests green"). Then a
quote card from `CONTRIBUTING.md`: "Fucina regresses in exactly two ways: it
becomes **wrong**, or it becomes **slow**."

**Overlay:** On the third failure mode: "real case: 7.2× self-speedup after
arming kernels that had been falling back to scalar — q8_0 pp256,
i9-13950HX (docs/BENCHMARK.md)".

### [0:24–0:53] An architecture you can check

**VO:** Catching wrong starts with structure. The tree is eleven bands —
dtypes at the bottom, models at the top — and a band may depend only on bands
at or below it. A checker in the tree runs Tarjan's algorithm over the
production import graph on every push: one hundred thirty-five files, five
hundred thirty-one edges, zero cycles. The gate states its own limit: cycles are
machine-checked; band direction is review-only — labeled, in writing.

**Visual:** Layer-stack diagram: eleven horizontal bands from
`docs/ARCHITECTURE.md`'s table (core · primitives · tensor · tags · backend ·
exec · tagged · ag+training · facade · llm · apps, bottom to top) with a
single downward arrow "may depend only on bands at or below". Then a terminal
shot: `zig build arch-check` running in the repo root. Close on a code shot:
`tools/check_import_graph.zig:1–15` (the header stating the contract,
including test-awareness and conservative parsing).

**Overlay:** "as recorded in docs/ARCHITECTURE.md: `production import graph:
135 files, 531 edges, 0 SCCs`" · "proves acyclicity, not band direction —
direction is review-checked (docs/DEVELOPMENT.md §1.1)".

### [0:53–1:40] The verification religion

**VO:** Then the religion. One sentence from the porting method: you don't
optimize what you can't verify — so the oracle is built first. An oracle is
any independent truth compared mechanically. Fucina keeps an arsenal.
Tokenizers: token-ID-exact against llama-tokenize. Model forwards: logit
parity against llama.cpp, from raw token ids, so a tokenizer bug can't
masquerade as a model bug. Quantization encoders: byte-exact against ggml.
Optimizers: goldens against their torch references. Gradients: finite
differences — calculus as a unit test. The deepest oracle is internal: a
scalar backend of plain loops, the executable specification every fast kernel
must agree with. Then the gates freeze into a ratchet: an optimization that
flips a single token is reverted, not tolerance-adjusted.

**Visual:** An inventory table builds row by row, synced to the VO (all rows
quoted from the chapter's §16.3 table): "tokenizers → token-ID-exact vs
`llama-tokenize`" · "forward passes → logit parity vs llama.cpp, from
explicit raw token ids" · "quant encoders → byte-exact vs ggml
(`src/backend/quant/encode_golden_test.zig`)" · "optimizers → torch-reference
goldens (`src/optim.zig`)" · "gradients → finite differences
(`src/ag/gradcheck.zig`)" · "fast kernels → the scalar backend,
`-Dbackend=scalar` (`src/backend/parity_test.zig`)". On the gradients row,
brief code shot: `src/ag/gradcheck.zig:1–12` (the contract-first doc
comment). Close on a quote card: "an optimization that flips a single token
is reverted, not tolerance-adjusted" (docs/PORTING.md).

**Overlay:** "you don't optimize what you can't verify — docs/PORTING.md" ·
on the ratchet card: "the tolerance is never loosened to make a gate pass".

### [1:40–2:00] Docs are tests

**VO:** Even documentation is on the hook. Docs lie predictably: true when
written, then the API moves. So CI runs snippet-check — every runnable block
in the reference manual is extracted, compiled against the real modules, and
executed on every push. Change the API, and the docs fail until they're
fixed.

**Visual:** Terminal shot: `zig build snippet-check` running in the repo
root. Beside it, a card listing the CI steps from the chapter's §16.5 (test ·
build · bench-check · arch-check · doc-check · snippet-check · x86dot-check ·
scalar leg · blas=none leg · llguidance leg), with `snippet-check`
highlighted.

**Overlay:** "docs/REFERENCE.md snippets run against the real
`fucina`/`fucina_llm` modules — in CI, every push (§2.7)".

### [2:00–2:40] A loss, kept and investigated

**VO:** Speed gets the same honesty. The paired gate runs both process
orders, gates on medians; rows noisier than eight percent are NOISY, not
counted. Losses print next to wins, dated. The record's most instructive
entry is a defeat: in the July fourth x86 snapshot, thirty-B mixture-of-experts
mid-batch prefill lost decisively — 0.36 to 0.52x. The loss was recorded with
a mechanism hypothesis. When the fix came, days later, the old path was
re-measured first — a control proving the comparison still held. Two levers,
measured separately. The residual, 0.965 to 0.987x on one band, stays on
the record: open, small.

**Visual:** Protocol card first (three bullets from `tools/bench_gate.py` via
§16.6: "both process orders · CV > 8% → NOISY, not a result · medians, exact
commands and raw output saved"). Then a five-step card sequence, one per VO
beat: "LOSS — pp15–33 band 0.36–0.52×" → "HYPOTHESIS — 128 experts routed
top-8: ~1–3 rows per expert, weight-bandwidth-bound" → "CONTROL — old path
re-measured, reproduced 0.375–0.509 (update dated 2026-07-10)" →
"DECOMPOSITION — phased-chain gate 512→64 did the heavy lifting; small-m
column chunking added the last few percent" → "RESIDUAL — Q6_K pp31–33 at
0.965–0.987×, 'open, small'".

**Overlay:** Persistent conditions caption under the entire sequence:
"Qwen3-30B MoE · i9-13950HX (Raptor Lake) · Linux · no BLAS either side ·
llama.cpp build 30af6e2 · snapshot 2026-07-04 · docs/BENCHMARK.md".

### [2:40–3:00] The transferable craft

**VO:** None of this needs Fucina's scale. The parity twin is a screenful of
code. The import gate is an afternoon. Recording losses as plainly as wins is
a decision, not an infrastructure. The craft is one habit: never let a claim
outrun its evidence. Next time: your forge.

**Visual:** Code shot: `src/x86dot_check.zig:17–33`, the execution-attestation
table, slow push-in with the two "NEVER" rows highlighted (aarch64 smmla,
x86 AVX512-VNNI) — the project saying NEVER about its own untested SIMD arms,
timed to land under "never let a claim outrun its evidence". End card: series
title, "Full chapter: `docs/course/16-the-craft.md`", "Next: 17 — Your
forge".

**Overlay:** On the attestation table: "dated, per-arm record of what has
actually *executed* — 'tested' never rounds up". End card: "Full chapter:
`docs/course/16-the-craft.md`" · "Next: Your forge".

## Asset list

**Code shots (repo files, exact ranges — verified in the current tree):**
- `tools/check_import_graph.zig:1–15` — the checker's header: production
  invariant, test-awareness, conservative parse fallback.
- `src/ag/gradcheck.zig:1–12` — the finite-difference referee's contract-first
  doc comment.
- `src/x86dot_check.zig:17–33` — the execution-attestation table; highlight
  the NEVER rows at lines 23 and 27.

**Terminal recordings (run in the repo root, no models needed):**
- `zig build arch-check` — the import-graph gate. Note: the live
  files/edges count may differ from the doc-recorded figure; the overlay
  quotes "135 files, 531 edges, 0 SCCs" explicitly *as recorded in*
  `docs/ARCHITECTURE.md`.
- `zig build snippet-check` — the REFERENCE.md runnable-snippet gate.

**Diagrams/cards to render (one sentence each):**
- Three quiet-failure text beats plus the two-ways quote card, text verbatim
  from `CONTRIBUTING.md` via chapter §16.0.
- Eleven-band layer stack from the `docs/ARCHITECTURE.md` table quoted in
  §16.1, single downward "may depend only on bands at or below it" arrow.
- Oracle-inventory table, rows quoted from the §16.3 arsenal table.
- Ratchet quote card: "an optimization that flips a single token is reverted,
  not tolerance-adjusted" (docs/PORTING.md §7/§4).
- CI-steps card, the ten steps as listed in §16.5.
- Paired-gate protocol card (both orders, CV > 8% NOISY, medians, raw output
  saved) from §16.6.
- Five-card loss sequence (LOSS → HYPOTHESIS → CONTROL → DECOMPOSITION →
  RESIDUAL), all numbers and phrases from §16.6 "A loss, investigated".
- End card with "Full chapter: `docs/course/16-the-craft.md`" and
  next-episode teaser.

**External downloads:** none — no model weights; every number is quoted from
the repo's records (`docs/BENCHMARK.md`, `docs/ARCHITECTURE.md`), not
re-measured.

## Production notes

- **Tone:** confident, concrete, zero hype — and never apologetic. The loss
  segment is the episode's centerpiece and is delivered with the same energy
  as any win: the point is that the protocol *worked*. Per the series rules,
  no state-of-the-art or production-readiness claims; the chapter's own
  self-grade is "production-oriented core, not production-ready product".
- **Caveats are load-bearing and MUST NOT be cut:** (a) the conditions
  caption (Qwen3-30B MoE, i9-13950HX, Linux, no BLAS either side, llama.cpp
  build 30af6e2, snapshot 2026-07-04) must be on screen whenever 0.36–0.52×,
  0.375–0.509, or 0.965–0.987× is; (b) the 7.2× overlay must carry "q8_0
  pp256 · i9-13950HX · docs/BENCHMARK.md"; (c) "135 files, 531 edges, 0 SCCs"
  must stay labeled "as recorded in docs/ARCHITECTURE.md" — if the live
  `arch-check` output differs, show the live output unedited and let the
  overlay carry the recorded figure; (d) the residual card keeps the words
  "open, small".
- **The NEVER shot is VO-unanchored by design:** the attestation table lands
  under "never let a claim outrun its evidence" — keep that sync; without it
  the shot reads as a non sequitur.
- **If the cut runs long, trim in this order:** the CI-steps card's dwell
  (keep the VO), then the oracle table's row-by-row animation (land it as one
  frame), then the live `zig build arch-check` terminal (replace with a
  static card of the doc-recorded output line). Never trim the loss sequence,
  the conditions caption, or any caveat overlay.
- **Numbers appearing in the video and their sources (nothing else may be
  quantified):** 7.2× self-speedup, q8_0 pp256, i9-13950HX
  (`docs/BENCHMARK.md` via §16.0); 135/531/0 (`docs/ARCHITECTURE.md` via
  §16.2); eleven bands (count of the ARCHITECTURE.md band table quoted in
  §16.1); CV > 8% NOISY default (`tools/bench_gate.py` via §16.6);
  0.36–0.52× pp15–33 band, 128 experts / top-8 / ~1–3 rows per expert
  (hypothesis card), 0.375–0.509 reverted-gate control (update dated
  2026-07-10), phased-chain gate 512→64, Q6_K pp31–33 residual 0.965–0.987×
  "open, small" (`docs/BENCHMARK.md` via §16.6, snapshot 2026-07-04).
- "Days later" in the VO refers to the 2026-07-04 snapshot vs the 2026-07-10
  fix update — both dates are in the chapter; do not sharpen to a day count
  in the VO, the overlay carries the exact dates.
- The next-episode teaser line ("your forge") matches Video 17's title and
  must survive edits.
