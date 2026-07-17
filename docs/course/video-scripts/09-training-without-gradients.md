# Video 09 — Training without gradients (3:00)

*Series: Forging Deep Learning in Zig · Source: ../09-training-without-gradients.md*

## Logline

Evolution strategies train models with forward passes only — perturb the
parameters with Gaussian noise, score each candidate with any scalar reward,
step toward the winners — and the trick that makes it practical is that the
noise is never stored, only regenerated from seeds. The video runs the
self-verifying `es_spirals` demo from scratch to 100% accuracy, then states
the honest price on camera: ES needs many more iterations than backprop, and
its value is reach, not efficiency.

## Takeaways

1. ES learns from rewards alone: θ plus σ·noise, a plain forward pass per
   candidate, z-scored rewards, a fitness-weighted step — no derivative of
   the reward is ever taken, so the reward can be discontinuous, discrete,
   or a black box.
2. The scale trick: every noise element is a pure function of (seed,
   iteration, member, slot, element) through a counter-based RNG, so
   perturb, restore, and update *regenerate* it — O(1) memory beyond the
   parameters, and a checkpoint is just an iteration counter.
3. Honesty and provenance: ES needs many more iterations than backprop for
   the same movement — a mechanism showcase, not a convergence recipe — and
   Fucina reimplemented the algorithm from the paper (the reference code's
   license forbids porting), then verified it bitwise with the reference run
   as an oracle.

## Script

### [0:00–0:25] Delete the backward pass

**VO:** Delete the backward pass. No graph, no gradients, no optimizer
state — and the model still learns. The signal is a single scalar reward per
candidate, and a reward can be anything you can compute: a negative loss, an
accuracy, even a rule that checks whether a reply starts with "Ahoy!". This
is evolution strategies, and its best trick is about memory.

**Visual:** Diagram: Chapter 8's training stack drawn as three boxes —
"autograd graph", "gradients", "optimizer state" — struck out one by one as
the VO negates them, leaving only "θ" and an arrow labeled "one scalar
reward per candidate". On "Ahoy!", a small card quotes the rule-reward
example from §9.8: *starts with `Ahoy!`, ends with `matey.`* (from
`examples/es_finetune/main.zig`, described in the chapter).

**Overlay:** "forward passes only — no `backward()`, no graph, no optimizer
state (`src/es.zig`)".

### [0:25–0:56] The whole algorithm, four lines

**VO:** Here is the entire algorithm, from the module doc of es.zig. Draw a
population of Gaussian noise vectors. For each one, evaluate the model at
theta plus sigma times that noise — a plain forward pass returning one
reward. Z-score the rewards. Then step theta toward the noise that scored
above the mean, away from the noise that scored below. No derivative of the
reward is ever taken — it can be discontinuous, discrete, a black box.

**Visual:** Code shot: `src/es.zig:11–14` — the four-line algorithm inside
the module doc — with line-by-line highlights synced to the VO: `eps_n ~
N(0, I)`, then `R_n = reward(theta + sigma * eps_n)` ("forward passes
only"), then the z-score line, then the update line.

**Overlay:** "vanilla OpenAI-ES, the reference's simplifications kept —
paper: arXiv:2509.24372" · "reward: discontinuous, discrete, or a black
box".

### [0:56–1:37] Noise that is never stored (core idea)

**VO:** Now the memory bill. Each noise vector is the size of the model, and
the update needs all of them. Stored naively at the paper's scale —
population thirty, a 0.6-billion-parameter model — that's roughly
seventy-two gigabytes of noise per iteration. That's arithmetic, not a
benchmark. Fucina's answer: the noise never exists as data. Element j of
member n at iteration t is a pure function of its coordinates through a
counter-based RNG — so perturb, restore, and update simply regenerate it,
three times per iteration. O(1) memory beyond the parameters. And a
checkpoint is just an iteration counter — no optimizer file at all.

**Visual:** Memory-bill card first: "30 members × 0.6e9 params × 4 bytes ≈
**72 GB of noise** — for one iteration", labeled "arithmetic, not a
benchmark" (§9.2). Then code shot: `src/es.zig:545–553` — `memberSeed`, with
its doc comment "a pure function of (config.seed, iteration, member).
Checkpoint contract" highlighted. Then a diagram: the coordinate tuple
"(seed, iteration, member, slot, element)" feeding one function box that
emits a single noise value, with three return arrows labeled *perturb* /
*restore* / *update* — all regeneration, no storage.

**Overlay:** "noise = pure function of (seed, iteration, member, slot,
element)" · "O(1) memory beyond θ · resume = an iteration counter — no
`optimizer.fucina` in an ES checkpoint".

### [1:37–2:07] es_spirals, from scratch (showcase)

**VO:** Watch it work. es_spirals trains the two-spirals classifier from
scratch — same MLP, same data as chapter eight, but every tensor is a plain
constant. No gradient tracking anywhere. Four worker threads each own a full
replica; only scalar rewards cross threads. One command. The run is
self-verifying: it exits non-zero below ninety percent accuracy, and chance
is fifty. On an M1 Max it reaches one hundred percent in about fifteen
thousand iterations — around seventy-five seconds.

**Visual:** Brief code shot: `examples/es_spirals/main.zig:39–48` — the `Model`
struct whose doc comment says "as CONSTANTS: ES needs no gradients, so
nothing here is a variable". Then the segment's centerpiece, a terminal
recording: `zig build es-spirals -Doptimize=ReleaseFast` — capture the
"before: accuracy ~0.5" line, timelapse through the progress prints (the
example prints accuracy every 500 iterations), land on the final 100%
accuracy and the clean exit.

**Overlay:** "self-verifying: exit ≠ 0 unless accuracy ≥ `--target` (default
0.90) · chance = 0.50" · persistent caption on the terminal: "M1 Max ·
ReleaseFast · ~15k iterations · ~75 s (~5 ms/iter) · population 128 —
dated, machine-specific snapshot (docs/TRAINING.md §13)".

### [2:07–2:37] The honest price

**VO:** Now sit with that number. Fifteen thousand iterations, one hundred
twenty-eight candidates each — nearly two million evaluations for a task
backprop solves in two thousand gradient steps. The docs say it plainly: ES
needs many more iterations than backprop for the same movement — a mechanism
showcase, not a convergence recipe. What ES buys instead is reach: ternary
weights with no gradient to take, rule rewards with no derivative,
full-parameter fine-tuning with no backward memory.

**Visual:** Comparison diagram, two labeled bars: "backprop
(`examples/spirals/main.zig`): 2,000 gradient steps" vs "ES
(`examples/es_spirals/main.zig`): ~15,000 iterations × 128 members ≈ 2M member
evaluations" — the ES bar drawn dramatically longer. Then a three-item
"reach" card: *quantized/ternary weights (packed TQ2_0 genomes)* ·
*non-differentiable rule rewards ("Ahoy!")* · *no-backward-memory
full-parameter fine-tuning* (§9.8).

**Overlay:** Quote card: "ES needs MANY more iterations than backprop for
the same movement … a mechanism showcase, not a convergence recipe" —
docs/TRAINING.md §13.

### [2:37–3:00] Never ported, and what's next

**VO:** One last thing. The reference implementation carries a noncommercial
license, so Fucina never ported it. The algorithm was rewritten from the
paper's math, then verified against the reference run as an oracle — bitwise
equal on identical noise, three dtypes, both noise schemes. Clean room, then
proof. Next time: a neural network inside a guitar amp, in real time.

**Visual:** Code shot: `src/es.zig:1–6` — the module doc's licensing stance,
highlighting "reimplemented here from the paper, not ported: the reference
code is under a noncommercial Academic Public License". Then a parity card:
"`tools/check_es_parity.py` — the reference (es-at-scale) executed as an
oracle vs a torch transcription of es.zig's algebra, same noise → bitwise
`torch.equal` on f32 / f16 / bf16, both noise schemes" (§9.7). End card:
series title, "Next: 10 — The guitar amp", chapter link
`docs/course/09-training-without-gradients.md`.

**Overlay:** "compared against, never ported (docs/TRAINING.md §13)" · end
card: "Next: The guitar amp — real-time neural audio".

## Asset list

**Code shots (repo files, exact ranges — all verified against the tree):**
- `src/es.zig:11–14` — the four-line ES algorithm in the module doc (frame
  may include lines 8–16 for context).
- `src/es.zig:545–553` — `memberSeed`: noise seed as a pure function of
  (config.seed, iteration, member), "Checkpoint contract" doc comment.
- `src/es.zig:1–6` — module doc licensing stance ("reimplemented … not
  ported").
- `examples/es_spirals/main.zig:39–48` — the `Model` struct of plain constants.
- Optional spare: `examples/es_spirals/main.zig:322–324` — the two-line training
  loop (`evaluateMembers` + `update`), if the showcase segment needs a beat
  between code and terminal.

**Terminal recording (executed on camera):**
- `zig build es-spirals -Doptimize=ReleaseFast` — from the repo root. Runs
  ~75 s on an M1 Max (machine-dependent); timelapse the middle. Must show:
  the "before: accuracy" line near 0.5, several of the every-500-iterations
  progress prints, the final accuracy, and the successful (zero) exit. The
  demo generates its own data — no downloads, no setup.

**Diagrams to render (one sentence each):**
- Backprop-baggage strikeout: graph / gradients / optimizer-state boxes
  struck out, leaving θ + "one scalar reward per candidate" (§9.0–9.1).
- Rule-reward card: *starts with `Ahoy!`, ends with `matey.`* (§9.8).
- Memory-bill card: 30 × 0.6e9 × 4 bytes ≈ 72 GB per iteration, labeled
  "arithmetic, not a benchmark" (§9.2).
- Noise-coordinates diagram: (seed, iteration, member, slot, element) → one
  function → one value; three arrows *perturb/restore/update*, all labeled
  "regenerate" (§9.2).
- Iteration-cost comparison bars: 2,000 gradient steps vs ~15,000 × 128 ≈ 2M
  member evaluations (§9.6).
- Reach card: ternary genomes · rule rewards · no-backward-memory
  full-parameter fine-tuning (§9.8).
- Parity card: reference-as-oracle vs torch transcription of es.zig's
  algebra, bitwise `torch.equal`, f32/f16/bf16, both noise schemes (§9.7).
- End card with next-episode teaser.

**External downloads:** none — `es_spirals` generates its own spiral data;
no model weights are needed anywhere in this episode.

## Production notes

- **Tone:** the honesty segment is the emotional center — deliver "nearly
  two million evaluations" with the same confident energy as the 100%
  accuracy result. It is a property of the method, not an apology; the very
  next breath is what ES *buys*. Never self-demoting, never overclaiming.
- **Caveats are load-bearing and MUST NOT be cut:** (a) the 72 GB figure
  must carry "arithmetic, not a benchmark" on screen; (b) the ~15k
  iterations / ~75 s / ~5 ms/iter / population-128 numbers must carry
  "M1 Max · ReleaseFast · dated, machine-specific snapshot
  (docs/TRAINING.md §13)" whenever visible; (c) the "MANY more iterations …
  mechanism showcase, not a convergence recipe" quote keeps its source
  (docs/TRAINING.md §13 — stated there about the `es-finetune` demo
  defaults; the chapter calls it "this chapter's required reading");
  (d) the parity claim must keep its exact shape — the reference executed
  as an oracle against a *torch transcription of es.zig's algebra* on
  identical noise — do not shorten it to "es.zig matches the reference
  bitwise".
- **The terminal run is machine-dependent:** ~75 s is the documented M1 Max
  figure; on other hardware record whatever it takes and keep the overlay's
  documented numbers unchanged (they quote the repo's snapshot, not the
  recording). The run is self-verifying — if it exits non-zero, that take is
  discarded, not narrated around.
- **Licensing wording must survive edits verbatim in spirit:**
  "reimplemented from the paper, not ported" — never "based on the
  reference code". This is a legal stance, not stylistic color.
- **If the cut runs long, trim in this order:** the `Model`-struct code shot
  in the showcase (the terminal carries the segment), then the rule-reward
  "Ahoy!" card in the hook (keep the VO line), then the algorithm shot's
  line-by-line highlight animation (land it as one frame). Never trim the
  honesty segment, the 72 GB caveat, or the licensing beat.
- **Numbers appearing in the video and their sources:** 72 GB = 30 ×
  0.6e9 × 4 bytes (§9.2, arithmetic); ~15k iterations, ~75 s, ~5 ms/iter,
  population 128, 100% accuracy (docs/TRAINING.md:982–988 via §9.6, M1 Max
  ReleaseFast); 2,000 gradient steps (`examples/spirals/main.zig:26` via §9.6);
  "nearly two million member evaluations" (§9.6's own arithmetic); target
  0.90 / chance 0.50 (§9.6); 4,482 parameters and 4 default workers (§9.6,
  VO says "four worker threads" — the chapter's stated default); three
  dtypes / both noise schemes (§9.7). Nothing else may be quantified.
- The next-episode teaser ("The guitar amp — real-time neural audio")
  matches Video 10's chapter title and must survive edits.
