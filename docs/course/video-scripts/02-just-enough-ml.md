# Video 02 — Just enough machine learning (3:00)

*Series: Forging Deep Learning in Zig · Source: ../02-just-enough-ml.md*

## Logline

The whole field in three sentences — a model is a function with knobs, a loss
measures wrongness, the derivative says which way to turn — demonstrated on a
one-knob model whose every number is asserted by compile-checked `zig test`
blocks, then pointed at the course's fruit fly: the two-spirals task.

## Takeaways

1. Training is walking downhill on the loss landscape: `w ← w − lr · dL/dw`.
   The same rule, unchanged except in scale, trains models with billions of
   parameters.
2. The course's arithmetic checks itself: `zig test` asserts both the
   convergence and the `lr = 0.3` divergence. The textbook is
   compile-checked.
3. Two spirals is the course's model organism — impossible for a linear
   model, plottable, trained in seconds — and its `trainStep` is the chapter's
   five-line loop grown up.

## Script

### [0:00–0:26] The whole field in three sentences

**VO:** Here is the entire field of machine learning in three sentences. A
model is an ordinary function with adjustable knobs. A loss is a single number
measuring how wrong its output is. Training turns each knob a little, in
whichever direction makes that number smaller — over and over, until it stops
shrinking. Everything else is machinery for doing that at scale.

**Visual:** Dark title card. The three sentences appear one at a time as
numbered lines (text verbatim from the chapter's opening list,
`docs/course/02-just-enough-ml.md` lines 14–17). Series mark small in a
corner.

**Overlay:** "Forging Deep Learning in Zig — 02 · Just enough machine
learning"

### [0:26–1:02] One knob, one parabola

**VO:** Take the smallest possible model: predict of x equals w times x. One
knob. One observation: when x is two, the answer is six. You can solve it in
your head — w should be three. We start it wrong, at one, and score it with
squared error: prediction minus target, squared. Plot that loss against the
knob and you get a parabola with its bottom at w equals three. That picture is
the loss landscape, and training is nothing more than walking downhill on it.

**Visual:** One diagram: the parabola `L(w) = (2w − 6)²` plotted over w from
0 to 6, minimum marked at w = 3, a dot sitting high on the left slope at
w = 1; a small inset shows `predict(x) = w · x` and the single data point
`x = 2 → y = 6`.

**Overlay:** "L(w) = (2w − 6)²" · "start: w = 1 · target: w = 3"

### [1:02–1:42] Downhill by derivative

**VO:** Which way is downhill? The derivative says. At w equals one the slope
is minus sixteen: the sign says increase w, the magnitude says the ground is
steep. That gives the entire update rule of deep learning — w becomes w minus
a learning rate times the slope. Run it with learning rate zero point one and
watch the loss: sixteen, zero point six four, zero point zero two five six.
The steps shrink on their own, because the slope shrinks near the bottom —
nobody schedules that. The same rule trains models with billions of
parameters.

**Visual:** The same parabola; the dot hops downhill step by step. Beside it,
one table builds one row at a time — the chapter's §2.3 descent table, steps
0–3: w 1.0 → 2.6 → 2.92 → 2.984, loss 16 → 0.64 → 0.0256 → 0.001024, slope
−16 → −3.2 → −0.64 → −0.128.

**Overlay:** "w ← w − lr · dL/dw" · small caption under the table: "course
math — verified by the chapter's zig test (§2.4)"

### [1:42–2:10] The textbook checks itself

**VO:** You don't have to trust that table. In this course the arithmetic is
asserted: the whole descent is a Zig test — no library, no allocator, nothing
but arithmetic — and zig test proves the loss falls below ten to the minus six
while w lands on three. A second test cranks the learning rate to zero point
three and asserts the divergence. The textbook checks itself.

**Visual:** Code shot: `one_knob.zig` — the chapter's course-code block,
extracted verbatim from `docs/course/02-just-enough-ml.md` lines 206–235
(both tests). Highlight in sequence: the five commented loop-body lines of
test "one knob, learned by gradient descent", then the two `try
std.testing...` asserts, then the second test's `w -= 0.3 * grad;` and its
`@abs(w - 3.0) > 2.0` assert. Cut to terminal recording: `zig test
one_knob.zig` (pinned toolchain, Zig 0.16.0) — tests pass.

**Overlay:** on the terminal shot: "the textbook checks itself"

### [2:10–2:42] Two spirals, the course fruit fly

**VO:** Meet the course's fruit fly: two interleaved spirals, four hundred
points, and one question — which arm does this point belong to? No straight
line can separate them; that is exactly the point. A small network — four
thousand four hundred eighty-two knobs — learns it in seconds on a laptop CPU,
and its training step is your five-line loop grown up: forward, loss,
backward, step, zero the gradients. One command runs it.

**Visual:** One diagram: scatter plot of the two spiral arms in two colors
(points generated from the `makeSpirals` math, `examples/spirals.zig:168–186`,
seed 1234). Then code shot: `trainStep`, `examples/spirals.zig:144–153`, its
five body calls highlighted in the order the VO names them. Then terminal
recording: `zig build spirals -Doptimize=ReleaseFast` from the repo root,
loss and accuracy lines scrolling.

**Overlay:** "MLP 2-64-64-2 (tanh) · 4,482 parameters · full batch" · on the
terminal shot: "seconds on a laptop CPU — machine-dependent"

### [2:42–3:00] Nothing stays magic

**VO:** Nothing here stays magic. Every operation in that training step gets
built in front of you, from an empty file, with tests proving it right. Next
time: the tensor — the one data structure that holds everything — built from
scratch.

**Visual:** Card: three rows from the chapter's §2.10 road-ahead table
("tensors as shape + numbers → Chapter 3", "`backward()` — the chain rule,
automated → Chapter 7", "`opt.step()`, learning rates, checkpoints →
Chapter 8") fading into the next-episode title card.

**Overlay:** "Next: 03 — Tensors from scratch"

## Asset list

- **`one_knob.zig`** — NOT a checked-in repo file: extract the course-code
  block verbatim from `docs/course/02-just-enough-ml.md` lines 206–235 (the
  two tests, including `const std = @import("std");`) into a scratch file
  named `one_knob.zig`. It must compile and pass as-is.
- **Terminal recording 1:** `zig test one_knob.zig` with the pinned toolchain
  (Zig 0.16.0), showing the passing run.
- **Code shot:** `examples/spirals.zig` lines 144–153 (`trainStep`).
  Optional b-roll if a beat needs filling: lines 129–142 (`forwardLogits`).
- **Terminal recording 2:** `zig build spirals -Doptimize=ReleaseFast` from
  the repo root (the file's own header recommends this invocation,
  `examples/spirals.zig:14`). No model download needed — the demo generates
  its own data.
- **Diagram 1:** parabola `L(w) = (2w − 6)²` with minimum at w = 3 and an
  animated dot descending from w = 1.
- **Diagram 2 (table):** the §2.3 descent table, steps 0–3, rows revealed one
  at a time (numbers exactly as in the chapter).
- **Diagram 3:** two-spirals scatter in two colors, generated from the
  `makeSpirals` math (`examples/spirals.zig:168–186`: θ = 3.5π·t,
  r = 0.15 + 0.85·t, class 1 = class 0 negated, noise σ ≈ 0.02, seed 1234,
  400 points).
- **Text cards:** the chapter's three-sentence opening list (lines 14–17);
  three rows of the §2.10 road-ahead table.

## Production notes

- **Pacing:** 426 VO words over 3:00 ≈ 142 wpm — mid-range. Segments 3 and
  4 carry the table rows and test asserts; let them land on their VO beats
  rather than rushing the narration.
- **Caveats that MUST stay attached to numbers:** The descent
  table is course math, verified by the chapter's §2.4 test — keep the
  "course math — verified by zig test" caption. "Seconds on a laptop CPU" is
  the chapter's qualitative framing — keep the "machine-dependent" overlay
  and never show or imply a specific wall-clock time.
- **Trim order if the cut runs long:** first the "You can solve it in your
  head" clause in segment 2; then the "The steps shrink on their own …
  nobody schedules that" sentence in segment 3 (its idea survives in the
  table).
- **Must not change:** the three-sentence opening (it is the chapter's own
  framing); the assertion thresholds (loss < 1e-6, w within 1e-3 of 3.0,
  |w − 3| > 2 after divergence); the lr values 0.1 and 0.3; the "textbook
  checks itself" beat — it is this episode's showcase.
- **Code integrity:** the `one_knob.zig` extraction must be byte-verbatim
  from the chapter. If the tests fail under the pinned Zig 0.16.0, stop and
  report upstream — do not edit the code to pass. (Verified 2026-07-12: the
  lines 206–235 extraction compiles and both tests pass with Zig 0.16.0.)
- **Tone:** warm, direct, no hype. No state-of-the-art or
  production-readiness claims; the spirals run's honest charm is
  reproducibility and smallness, not speed records.
- **Spirals terminal shot:** the demo prints five optimizers plus
  checkpoint-resume lines; any crop should show loss values falling and an
  accuracy line, without implying a runtime figure.
