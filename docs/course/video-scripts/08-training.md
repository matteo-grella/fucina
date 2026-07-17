# Video 08 — Training: making the machine learn (3:00)

*Series: Forging Deep Learning in Zig · Source: ../08-training.md*

## Logline

A gradient is only a direction — training is the fixed six-stage ritual that
turns it into learning, and an optimizer is a progression from a one-line SGD
to Muon and APOLLO, each a golden-parity port of its reference. The showcase
runs on camera: `zig build spirals`, five optimizers through a
checkpoint-and-resume gauntlet whose pass condition is bit-for-bit equality —
`error.ResumeNotBitExact` if a single bit flips.

## Takeaways

1. A training step is a fixed six-stage ritual — forward → loss → `backward`
   → `clipGradNorm` → `step` → `zeroGrad` — canonical order, pinned by tests;
   and skipping `zeroGrad` isn't a bug, it *is* gradient accumulation.
2. Optimizers are a progression — SGD is one line, AdamW is twelve, Muon and
   APOLLO exploit matrix structure in opposite directions — and every one in
   the repo is a golden-parity port of its actual reference, quirks included.
3. Determinism is a testable property: the spirals demo resumes from a
   halfway checkpoint into a fresh model and demands every final parameter be
   *identical* — `!= 0` fails the program, not a tolerance check.

## Script

### [0:00–0:22] A direction is not learning

**VO:** The last episode ended with a superpower: call backward, and every
parameter's gradient appears. But a gradient is only a direction. Nothing has
learned anything yet. Training is what turns direction into learning — and
here it is not a framework. It is a for loop you write yourself, with a fixed
six-stage ritual inside.

**Visual:** Full-screen code shot of the canonical training step,
`docs/TRAINING.md:16–42` (the same snippet the chapter opens §8.1 with),
typing in line by line; the camera settles on the `for (0..total_steps)`
loop as the VO says "for loop".

**Overlay:** "no trainer framework, no callbacks — a `for` loop you write
yourself".

### [0:22–0:53] The six-stage ritual

**VO:** Here is the whole ritual. Forward: ordinary op calls — the graph
builds itself. Loss: one more op, reducing the batch to a scalar. Backward:
it adds each gradient into a persisted accumulator. Clip: it must see
complete gradients — after backward, before step. Step: the optimizer reads
the gradients and updates the parameters. Zero-grad: clear, done. That order
is canonical, and the repo pins it with tests. And skipping zero-grad isn't a
bug — it is gradient accumulation.

**Visual:** Same code shot, six sequential highlights synced to the VO:
(1) the forward lines (`dot`/`add`/`tanh`), (2) the `crossEntropy` line,
(3) `loss.backward`, (4) `clipGradNorm` with its comment "after backward,
before step", (5) `opt.step`, (6) `opt.zeroGrad()`. Stage numbers 1–6 tick
on beside each highlight.

**Overlay:** "canonical order: `backward` → `clipGradNorm` → `step` →
`zeroGrad` (docs/REFERENCE.md §11, pinned by tests)" · at the last beat:
"skip `zeroGrad` on purpose = gradient accumulation (ch. 8, §8.8)".

### [0:53–1:23] One line, then twelve

**VO:** Now the optimizer, as a progression. Stochastic gradient descent is
genuinely one line: move each parameter a small step against its gradient.
That one-liner already trains today's demo to convergence. AdamW, the
workhorse, is twelve lines: decoupled weight decay, two moving averages, a
bias-corrected step — the scalars prepared once per step in sixty-four-bit
floats. And these are not "roughly Adam": every optimizer here is a
golden-parity port, quirks included, tested against its actual reference
implementation.

**Visual:** Code shot 1: the course-code SGD one-liner,
`docs/course/08-training.md:204–208`, kept on screen with its "Course code —
NOT from the Fucina repo" comment visible. Code shot 2: the shipping AdamW
inner loop, `src/optim.zig:756–767` (`runScalar`), with three highlights in
sequence: `decayed` (decoupled decay), the `m1`/`v1` lines (the two EMAs),
the final line (bias-corrected step). Then a small card: "golden parity:
PyTorch 2.12 · Keller Jordan's muon.py · apollo_torch — goldens in
`src/optim_tests.zig`".

**Overlay:** "SGD: 1 line · AdamW: 12 lines" · "bias corrections computed
once per step, in f64 (docs/REFERENCE.md §11)".

### [1:23–1:53] The frontier, and what it costs

**VO:** Past the workhorse, the frontier — in two directions. Muon treats a
weight matrix as a matrix: it orthogonalizes the momentum with Newton-Schulz
iterations and takes better steps, paying real matrix-multiply flops for it.
APOLLO goes the other way: Adam's moments live in a compressed space, behind
a random projection regenerated from a seed instead of stored — attacking
optimizer state, which at LLM scale costs more than the model. The bench
table makes the trade concrete.

**Visual:** Brief scroll of `src/optim.zig:1384–1416` (`newtonSchulz5` with
its doc comment "approximates U*V^T with singular values in roughly
(0.5, 1.5) — by design, not a bug"). Then a rendered table card, one
15.7M-param transformer block, ms/step: sgd 1.8–3.0 · adamw 5.2–6.0 ·
apollo-mini 23–24 · apollo-r256 44–46 · muon 446–457 (docs/TRAINING.md §11).

**Overlay:** Persistent caption under the table: "M1 Max · 2026-06-10 ·
native backend + Accelerate · `zig build bench-optim -Doptimize=ReleaseFast`
· dated, machine-specific snapshot — not a general claim". Quote card:
"When someone tells you optimizer choice is free, this table is the
counterexample." (ch. 8, §8.6).

### [1:53–2:18] On camera: zig build spirals

**VO:** Time to collect. One command: zig build spirals. Four hundred
ninety-three lines: two interleaved spiral arms, a small MLP, five
optimizers, each run through a three-phase gauntlet — train two thousand
steps, checkpoint halfway, resume, then pure inference from the checkpoint.
Watch the loss column: every optimizer drives it down, at visibly different
speeds and depths. The whole optimizer story, rendered as data.

**Visual:** Live terminal recording:
`zig build spirals -Doptimize=ReleaseFast`. The header line ("two spirals:
… points, MLP 2-…-…-2 (tanh), full-batch, … steps") appears, then the
per-optimizer blocks print as each gauntlet finishes: `[sgd]`, `[adamw]`,
`[muon]`, `[apollo]`, `[apollo-mini]`, then the groups demo. Highlight
sweeps down the loss values on the "trained … steps: loss …" lines as the
VO says "loss column". The printed numbers are whatever the recording
machine produces — real output only.

**Overlay:** "`examples/spirals/main.zig` — 493 lines, the whole chapter
runnable" · "no downloads: the demo generates its own 400 points".

### [2:18–2:45] The gate: != 0

**VO:** Now the resume line. Phase two builds a fresh model from a different
seed, loads the checkpoint, retrains the second thousand steps, and compares
every final parameter against the run that never stopped. That check is not
a tolerance. It is: not equal zero. One flipped bit anywhere, and the
program fails with error resume-not-bit-exact. Determinism here is a
testable property — so it is tested.

**Visual:** Split screen. Left: the same terminal, highlighting each
"resume from step 1000: max |delta param| = 0 (bit-exact)" line as it
appears — six of them across the run. Right: code shot
`examples/spirals/main.zig:309–314` — the `max_diff` loop and
`if (max_diff != 0) return error.ResumeNotBitExact;` — with a second brief
cut to line 298 (`Model.initRandom(ctx, 7)` — "different init: fully
overwritten by the checkpoint").

**Overlay:** "`if (max_diff != 0) return error.ResumeNotBitExact;` — not a
tolerance" · "fresh model, different seed — nothing can leak".

### [2:45–3:00] Why it matters, and what's next

**VO:** Bit-exact resume is the repo's debugging instrument: when it holds,
any divergence between runs is a change you made. Schedules, mixed
precision, accumulation — the chapter has the rest. Next time: training
without gradients at all.

**Visual:** Hold the terminal's final frame for a beat, then end card:
series title, "Next: 09 — Training without gradients", chapter link
`docs/course/08-training.md`.

**Overlay:** End card: "Next: Training without gradients".

## Asset list

**Code shots (repo files, exact ranges):**
- `docs/TRAINING.md:16–42` — the canonical training-step snippet (§1),
  used for both the hook and the six-stage highlight pass.
- `docs/course/08-training.md:204–208` — the course-code `sgdStep`
  one-liner (compile-checked course code; keep its "Course code — NOT from
  the Fucina repo" comment in frame).
- `src/optim.zig:756–767` — the AdamW `runScalar` per-element update.
- `src/optim.zig:1384–1416` — `newtonSchulz5` with its doc comment.
- `examples/spirals/main.zig:309–314` — the bit-exact gate; plus line 298 (the
  fresh model from seed 7).

**Terminal recordings (execute on camera):**
- `zig build spirals -Doptimize=ReleaseFast` — the episode's centerpiece.
  Record the full run once; time-compress dead air in the edit but never
  alter the printed text or numbers. Expected shape of the output: one
  header line, then three lines per optimizer for `[sgd]`, `[adamw]`,
  `[muon]`, `[apollo]`, `[apollo-mini]`, then two lines for the groups demo,
  labeled `[adamw-groups]` (format strings at
  `examples/spirals/main.zig:291–321,347` and, for the groups demo, `:452,487`;
  six "bit-exact" resume lines total). Note: the demo prints one summary
  loss per optimizer
  at the end of its 2000 steps — there is no per-step loss stream — so the
  "loss column" beat highlights the loss values across the accumulated
  optimizer lines. Pacing datum (not for on-screen use): a dry run on
  2026-07-12 (M1-class machine, warm build cache) completed in ~21 s wall
  clock including the build, so the run fits the terminal segments with
  little or no time compression.

**Diagrams / cards to render (one sentence each):**
- Golden-parity card: "PyTorch 2.12 · Keller Jordan's muon.py ·
  apollo_torch — goldens in `src/optim_tests.zig`" (docs/TRAINING.md §3).
- Optimizer cost table: ms/step for one 15.7M-param transformer block —
  sgd 1.8–3.0 · adamw 5.2–6.0 · apollo-mini 23–24 · apollo-r256 44–46 ·
  muon 446–457 — with its mandatory caveat caption (docs/TRAINING.md §11).
- Quote card: "When someone tells you optimizer choice is free, this table
  is the counterexample." (chapter §8.6).
- End card with the next-episode teaser.

**External downloads:** none — the spirals demo generates its own dataset
(400 points, in-program); no model weights are needed anywhere in this
episode.

## Production notes

- **Tone:** warm, concrete, quietly confident. The bit-exact gate is the
  emotional peak — deliver "not equal zero" with weight, not hype. Never
  frame the one-person-with-agentic-assistance context apologetically if it
  comes up; this episode doesn't need it in VO.
- **Do not fabricate numbers.** The spirals losses, accuracies, and the
  `max |delta param| = 0` lines come from the actual on-camera run. If a
  recorded run ever printed "NOT bit-exact", that is a stop-ship bug to
  report, not something to edit around.
- **Caveats are load-bearing and MUST NOT be cut:** the optimizer cost
  table must carry "M1 Max · 2026-06-10 · native backend + Accelerate ·
  dated, machine-specific snapshot — not a general claim" whenever any of
  its numbers is on screen. The `sgdStep` shot must keep its "Course code —
  NOT from the Fucina repo" marker. "PyTorch 2.12 / muon.py / apollo_torch"
  is quoted from docs/TRAINING.md §3 and must not drift.
- **Numbers appearing in the video and their sources:** six stages and the
  canonical order (chapter §8.1, docs/REFERENCE.md §11); "one line" SGD and
  "twelve lines" AdamW (chapter §8.4–8.5); f64 scalar prep (chapter §8.5);
  optimizer ms/step table (docs/TRAINING.md §11 via chapter §8.6); 493
  lines, 2000 steps, checkpoint at 1000, 400 points, seed-7 fresh model,
  six bit-exact lines (chapter §8.11 / `examples/spirals/main.zig`); "state
  costs more than the model at LLM scale" (chapter §8.6: AdamW state is
  8 bytes/param). Nothing else may be quantified.
- **If the cut runs long, trim in this order:** the `newtonSchulz5` scroll
  (keep the VO and the table card), then the golden-parity card's dwell
  (keep the VO sentence), then shorten the six-highlight pass to three
  highlights (backward → clip → step). Never trim the cost-table caveat,
  the terminal run, or the `!= 0` code shot.
- The next-episode teaser line ("Training without gradients") matches
  Video 09's title and must survive edits.
