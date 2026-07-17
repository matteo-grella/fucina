# Video 17 — Your forge (3:00)

*Series: Forging Deep Learning in Zig · Source: ../17-epilogue.md*

## Logline

The series finale: the whole seventeen-chapter journey replayed in sixty
seconds — from a dtype enum to a live guitar amp and a chatting transformer,
one language, one machine — then the hammer changes hands: the chapter's
graded projects, the oracle-first method behind all of them, and the forge
the name promised.

## Takeaways

1. The arc is real and readable end to end: dtypes → tensors → typed axes →
   SIMD ops → autograd → a real-time amp → GGUF, a transformer, a GPT
   trained from scratch — one program, one language, one machine.
2. The transferable method: in every project, the first milestone is the
   thing that can tell you you are wrong. The oracle comes first. And a
   library regresses in exactly two ways — wrong, or slow.
3. Your turn is documented, not hypothetical: six graded projects, each with
   a path in the repo, from `zig build spirals` (no downloads) to a PR where
   "line-by-line authorship is not the bar — accountability is."

## Script

### [0:00–0:14] Hook: it began with an enum

**VO:** Seventeen episodes ago, this series made a promise: a modern
deep-learning stack — the part that hides below Python — as one program, in
one language, you can read top to bottom. It began with an enum.

**Visual:** Slow push-in on `src/dtype.zig:3-16` — the `DType` enum, from
`bool` and the integer types down through `f16`/`bf16`/`f32` into the first
quantized cases. No terminal, no diagram: just the enum that started
Chapter 3.

**Overlay:** "`src/dtype.zig` — where Chapter 3 began".

### [0:14–0:54] The journey, in sixty seconds

**VO:** A dtype enum, a refcounted buffer, stride arithmetic — a tensor.
Then axis names moved into the type system, and a misaligned contraction
became a compile error. An op library; SIMD kernels held to a scalar twin.
Autograd with no tape, no graph object — the live tensors are the graph.
Then the library met the world: a WaveNet ran a guitar amp live, from a
callback that never allocates. And the climb to language models — GGUF,
quantization, a transformer, a GPT trained from scratch on your own
machine. One language. One machine. Nothing you cannot read.

**Visual:** Montage in five beats, cut on the VO phrases: (1) quick code
flicker across `src/storage.zig` and `src/tensor.zig` file headers; (2) the
journey map — the §17.1 review-index table drawn as a rising path, Part I
through Part VI, each station labeled with its repo path (`src/dtype.zig`,
`src/tags.zig`, `src/backend/`, `src/ag/`, `examples/nam/`, `src/llm/`,
`docs/`); (3) live-amp clip reused from Video 10 (guitar → USB interface →
terminal, playing); (4) terminal recording: `zig build qwen3
-Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --repl`
(`examples/qwen3/README.md`), a short question streaming an answer;
(5) hard cut to a black card as the VO lands the last three sentences.

**Overlay:** On the black card, the chapter's own line: "From a dtype enum
to a live guitar amp and a chatting transformer. One language. One machine.
Nothing you cannot read." (§17.1).

### [0:54–1:36] Your turn: the oracle comes first

**VO:** Now the hammer changes hands. The chapter grades six projects, and
in every one the first milestone is the thing that can tell you you are
wrong. The oracle comes first — the whole course, applied to your work.
Start tonight: spirals needs no downloads, and RUNNING-MODELS has
copy-paste commands for the model zoo. Build ReleaseFast — Debug is ten to
fifty times slower. Then add a pointwise op end to end: prototype it in an
evening with elementalUnary, or go the full route, where the compiler walks
you through — add the enum variant, and every site that must learn about it
becomes a compile error until it does.

**Visual:** The six-project ladder — the §17.2 table rendered as a graded
ramp (1 run every example … 6 contribute), with the "first milestone"
column highlighted as the VO says "the thing that can tell you you are
wrong". Then terminal recording: `zig build spirals -Doptimize=ReleaseFast`
running to a decreasing loss. Then code shot `src/ag/tensor.zig:1559-1570`
(the `elementalUnary` doc comment and signature), followed by the
five-station route diagram for the full route: `src/backend/ops.zig`
(enum + switch) → `src/backend/vector/primitives.zig:155` (`vecUnary`) →
`src/exec/elementwise.zig` → `src/ag/backward.zig` → `src/ag/tensor.zig`,
with `src/backend/ops.zig:40-57` (`UnaryOp`, `tanh` at line 56) shown
beside the first station.

**Overlay:** "ReleaseFast is mandatory — Debug is 10–50× slower
(README.md, via §17.2)" · on the route diagram: "no `else` in the switch —
'the compile error is the enforcement' (docs/DEVELOPMENT.md)".

### [1:36–2:05] Hard mode: port a model, capture your amp

**VO:** Ready for hard mode? Port a small model — and don't improvise the
method. PORTING.md is the method: pin the reference at an exact commit,
build the oracle first, close parity stage by stage, and never loosen a
tolerance to make a gate pass. Or capture your own amp: the NAM example
walks the whole path, and the trainer refuses clipped input. There is
something clarifying about a project whose loss you can hear.

**Visual:** Doc shot `docs/PORTING.md:1-10`, highlighting the one-line
version: "you don't optimize what you can't verify". A small parity-ladder
diagram climbs beside it: tokenizer (token-ID-exact) → logits from raw ids
→ generation. Then doc shot `examples/nam/README.md:163-170` — the one-step
`fucina-nam profile --signal v3_0_0.wav …` command and the two-step
`train --input … --output … --out my-amp.nam` line — over rig B-roll reused
from Video 10.

**Overlay:** "discrete outputs gate on exact equality; tolerances are never
loosened (docs/PORTING.md via §17.2)" · "the trainer refuses clipped
captures (`examples/nam/README.md`)".

### [2:05–2:31] Contribute: wrong or slow, and a human sends the PR

**VO:** And when you are ready, contribute. Two rules matter most. Fucina
regresses in exactly two ways — it becomes wrong, or it becomes slow — and
your change is tested against the failure modes it can realistically
affect. And a human sends the PR. Working with coding agents is expected —
the project is built that way — but line-by-line authorship is not the bar.
Accountability is.

**Visual:** Doc shot `CONTRIBUTING.md:23-27` ("Fucina regresses in exactly
two ways: it becomes **wrong**, or it becomes **slow**… A doc fix needs no
benchmark… a kernel change needs both tracks, always"), then
`CONTRIBUTING.md:8-19`, pull-quote highlight on "Line-by-line authorship is
not the bar — accountability is."

**Overlay:** "'Tests pass' without the machine and the commands is not a
report (§17.2)".

### [2:31–3:00] The forge is yours

**VO:** Fucina is Italian for forge, and the name was never decoration. A
forge is not a factory — it's a small, hot room where a person with
judgment applies heat and pressure, checks the work against a straightedge,
and hits it again. Every chapter was that loop. The stack is not a tower
you live under anymore. It's a workshop you can walk into. The fire is lit.
The forge is yours.

**Visual:** The forge-loop diagram, drawn as a cycle: HEAT (a real
application that needed something the library didn't have) → PRESSURE (the
benchmark, the profiler, the 48 kHz deadline) → STRAIGHTEDGE (an oracle: a
reference implementation, a finite-difference check, a scalar twin, a
golden byte) → the next hammer blow, looping back (§17.4). As the VO turns
personal, dissolve through three stills from the series — the enum, the
amp, the chat — dimming into the final end card: series title, the
chapter's closing line "The fire is lit. Forge something." and the link
`docs/course/17-epilogue.md`. No next-episode teaser — this is the finale.

**Overlay:** End card: "The fire is lit. Forge something." (§17.4) · "The
forge is yours." · "docs/course/17-epilogue.md".

## Asset list

**Terminal recordings:**
- `zig build spirals -Doptimize=ReleaseFast` — needs no downloads at all
  (§17.2, project 1); hold on the loss decreasing.
- `zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf
  --repl` — one short question, streamed answer (`examples/qwen3/README.md`).

**External downloads (weights are NOT in the repo):**
- `Qwen3-0.6B-Q8_0.gguf` from `Qwen/Qwen3-0.6B-GGUF` on Hugging Face
  (verified link table in `docs/RUNNING-MODELS.md:31`; each model family
  carries its own license — the doc notes terms next to each download).

**Reused footage (from Video 10's shoot):**
- Live-amp clip (guitar → interface → terminal) for montage beat 3 and the
  NAM rig B-roll in segment 5. If re-recording instead: 2–3 `.nam` profiles
  from Tone3000 (https://www.tone3000.com) and the Video 10 hardware list.

**Code/doc shots (repo files, exact ranges):**
- `src/dtype.zig:3-16` — the `DType` enum opening (hook).
- `src/storage.zig` and `src/tensor.zig` file headers (montage flicker; any
  top-of-file frame is fine).
- `src/ag/tensor.zig:1559-1570` — `elementalUnary` doc comment + signature.
- `src/backend/ops.zig:40-57` — `UnaryOp` enum (`tanh` at line 56).
- `docs/PORTING.md:1-10` — the one-line version of the method.
- `examples/nam/README.md:163-170` — `profile` one-step command + two-step
  `train` line.
- `CONTRIBUTING.md:8-19` — "A human sends the PR" section.
- `CONTRIBUTING.md:23-27` — the two regression tracks.

**Diagrams to render (one sentence each):**
- Journey map: the §17.1 review-index table as a rising path, Parts I–VI
  with their repo paths as station labels.
- Six-project ladder: the §17.2 table as a graded ramp with the
  first-milestone column highlighted.
- Five-station op route: ops.zig → vector/primitives.zig:155 →
  exec/elementwise.zig → ag/backward.zig → ag/tensor.zig.
- Parity ladder: tokenizer (token-ID-exact) → logits from raw ids →
  generation (§17.2, project 4).
- Forge loop: heat → pressure → straightedge → next hammer blow, annotated
  with the §17.4 mappings.
- Final end card (series close, no teaser).

## Production notes

- **This is the series finale.** The format's standing rule ("final segment
  ends with a one-line teaser") is deliberately overridden: no teaser, no
  "next time". The episode closes the series on "The forge is yours."
- **Tone:** valedictory but dry — the montage earns the emotion; the
  narration never inflates. No SOTA claims, no production-readiness claims.
  The contribution beat keeps the project's own candor: coding agents are
  expected, a human answers for the change.
- **Quotes are load-bearing:** "line-by-line authorship is not the bar —
  accountability is" (CONTRIBUTING.md:16-17), "you don't optimize what you
  can't verify" (docs/PORTING.md:6-7), "The fire is lit. Forge something."
  (§17.4) must appear verbatim, on screen or in VO, and keep their sources.
- **The only figure in this episode** is "Debug is 10–50× slower", quoted
  from the README via §17.2; its overlay citation stays attached. Add no
  other numbers — in particular, do not import performance figures from
  earlier episodes into the montage; the montage claims are qualitative by
  design.
- **Montage beats are reuse-first:** beat 3 and the segment-5 B-roll come
  from Video 10's footage. If a tags-compile-error clip from Video 04
  exists, it may replace part of montage beat 2 (the journey map), but the
  map must still appear — it is the episode's only whole-course visual.
- **If the cut runs long, trim in this order:** the parity-ladder
  mini-diagram (keep the PORTING.md doc shot), then the
  `src/backend/ops.zig` beside-shot on the route diagram, then shorten the
  spirals terminal hold — never the montage, never the forge loop, never a
  quote overlay.
- **Pacing note:** segment 3 (0:54–1:36) is the densest read (~107 words in
  42 s ≈ 153 wpm). If the voice needs air, steal 2 s from the spirals
  terminal hold rather than cutting VO.
