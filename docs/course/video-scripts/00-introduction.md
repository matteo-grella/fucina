# Video 00 — Why deep learning in Zig? (3:00)

*Series: Forging Deep Learning in Zig · Source: ../00-introduction.md*

## Logline

The mainstream deep-learning stack is a tower of languages you can't read
end to end; this series builds one you can — in Zig, top to bottom — and
proves the thesis with a ~30-line README program whose *same* forward
function both infers and trains.

## Takeaways

1. **The tower problem**: from Python to the silicon there are several
   codebases and languages between you and what actually runs — and your
   understanding stops at each layer.
2. **What you read is what runs**: Fucina is eager, one language top to
   bottom — and the same forward function serves inference *and* training,
   via exec scopes.
3. The series is honest by construction: one person's effort with strong
   agentic-coding assistance, no state-of-the-art claims, everything cited
   to file (usually with line numbers) — and it ends at a real-time guitar
   amp and a chatting transformer you understand completely.

## Script

### [0:00–0:25] The tower

**VO:** Open any modern deep-learning project and count the languages
between you and the silicon. You write Python. The Python calls a
framework. The framework dispatches into C++. The C++ launches CUDA
kernels, or hands your graph to a compiler you will never read. When the
model is wrong, or slow, you're debugging a tower — and you only own the
top floor.

**Visual:** Animated diagram: a vertical tower of floors labeled top to
bottom "your Python" → "framework" → "C++" → "CUDA kernels / vendor BLAS /
graph compiler", with only the top floor lit; the lower floors dim as each
is named.

**Overlay:** The tower problem — you only own the top floor.

### [0:25–0:55] The other path

**VO:** This series takes the other path: a deep-learning stack in one
language, top to bottom — Zig. The library is Fucina, Italian for forge.
Nothing sits underneath it but the standard library and your CPU. And it's
eager: every operation runs the moment you call it, on real buffers. No
graph to build, plan, or compile. The repo's README states the thesis in
one line: what you read is what runs — in inference and in training alike.

**Visual:** Code/text shot of the repo `README.md:3-9` (the opening
paragraph); a highlight sweeps across "what you read is what runs, in
inference and in training alike" as the VO reaches it.

**Overlay:** Fucina — Italian for *forge* · "what you read is what runs"

### [0:55–1:45] One screen of code, two jobs

**VO:** Here's the proof, from the front page of the README. About thirty
lines: a two-layer network, its forward pass, a training loop. Look at the
types — the axes have names. Tensor of batch and in is a different type
from in and batch, and a misaligned contraction is a compile error, not a
runtime shape crash three layers deep. Every intermediate dies at a
visible defer — you can see when each buffer is freed. And here is the
trick worth pausing on: this same forward function infers and trains. Open
an exec scope, and the identical code records what backward needs. No
train mode, no second graph. One function, both jobs.

**Visual:** Code shot of `README.md:24-53` (the front-page example), in
two passes: first frame the `Model` struct and `forward` (lines 24–41),
highlighting the axis-tag types `Tensor(.{ .batch, .in })` (lines 25–28
and 31) and the `defer z1.deinit()` lines; then pan down to the training
block (lines 43–53), highlighting `openExecScope` (line 48) and
`loss.backward` / `opt.step` (lines 52–53).

**Overlay:** (pass 1) axis names live in the types → shape errors at
*compile time* · (pass 2) the SAME forward infers and trains

### [1:45–2:12] Where this goes

**VO:** Where does the series go? Two destinations. One: a real-time
neural guitar amplifier — a WaveNet processing a live guitar signal inside
an audio callback, where one stray allocation is an audible glitch.
Fitting, because Zig itself was born from an audio project — Andrew
Kelley's Genesis. Two: a
transformer answering questions — Qwen3 chatting on your laptop, and you
will have read every stage under it: tokenizer, attention, sampling.

**Visual:** Course-map diagram: six parts, chapters 01→17 as stations on a
path, with two flagged destinations — Ch 10 "The guitar amp" (guitar +
waveform icon) and Ch 12 "A transformer from scratch" (chat-bubble icon).
At "Qwen3 chatting on your laptop", cut to a short terminal recording of
`zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf
--chat "What is the capital of France?" --no-think` (the README quick
start, `README.md:125-126`) printing its answer.

**Overlay:** Ch 10: a neural guitar amp, live · Ch 12: a transformer,
line by line

### [2:12–2:40] Cards on the table

**VO:** Now, plainly: Fucina is one person's effort, built with strong
assistance from agentic coding systems — humans leading the ideas, the
testing, and the debugging. It makes no state-of-the-art claims, and this
course itself was generated with AI over the library as it stands today.
That candor is a feature: every claim is cited to file, usually to the
line, and where text and source disagree, the source wins.

**Visual:** Text shot of `README.md:262-268` (the agentic-assistance
passage), highlight on "humans leading the ideas, the testing, and the
debugging"; then a brief shot of the course chapter
`docs/course/00-introduction.md` §0.7 ("How this course was made"),
zooming on a `path:line` citation to show the citation discipline.

**Overlay:** No state-of-the-art claims · every claim cited · if text and
source disagree, the source wins

### [2:40–3:00] Your forge

**VO:** Everything here runs on the machine you already own. Clone the
repo, run zig build test — no model files, no GPU, no cloud. When it
passes, you hold a working forge. Next up: just enough Zig to read all of
it.

**Visual:** Terminal recording: `git clone
https://github.com/matteo-grella/fucina && cd fucina && zig build test`
(the README quick start, `README.md:117-121`), ending on the passing test
summary. Hold on an end card: series title + "Next: 01 — Just enough Zig".

**Overlay:** `zig build test` — no model files needed · Next: 01 — Just
enough Zig

## Asset list

- **Code shots** (from the repo, as-is):
  - `README.md:3-9` — opening paragraph, "what you read is what runs".
  - `README.md:24-53` — the front-page example (Model struct, `forward`,
    exec-scope training block).
  - `README.md:262-268` — the agentic-assistance passage.
  - `docs/course/00-introduction.md` §0.7 — the "How this course was made"
    passage (brief, for the citation-discipline zoom).
- **Terminal recordings** (Zig 0.16.0 required — the toolchain is pinned;
  other versions will not build):
  - `git clone https://github.com/matteo-grella/fucina && cd fucina && zig
    build test` — needs no model files and no network beyond the clone.
  - `zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf
    --chat "What is the capital of France?" --no-think`
- **Model download** (weights are NOT in the repo; ~1 GB disk/network):
  `hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir
  models` (source: the README quick start, `README.md:122-126`).
- **Diagrams to draw**:
  - The tower: floors "your Python / framework / C++ / CUDA kernels,
    vendor BLAS, graph compiler", only the top floor lit.
  - The course map: six parts, chapters 01→17, flags on Ch 10 (guitar amp)
    and Ch 12 (transformer).
- **End card**: series title + "Next: 01 — Just enough Zig".

## Production notes

- **Tone**: warm, direct, unhurried; the candor segment is spoken as a
  point of pride, not an apology — honest, never self-demoting, never
  overclaiming.
- **Pacing**: segment 3 (the code) is the heart — let the highlights land
  with the VO beats; everything else can tighten around it.
- **Trim order if long**: first shorten the terminal beats in segment 4
  (the map diagram alone can carry it), then compress segment 1's tower
  animation. Do NOT trim segment 5 (the candor block) — it is required —
  and do not cut the teaser line.
- **Must not change**: the quote "what you read is what runs" verbatim;
  the candor facts (one person's effort, strong agentic-coding assistance,
  humans leading ideas/testing/debugging, course AI-generated over the
  library as it stands today, no state-of-the-art claim, source wins);
  code on screen is the repo's own, never mocked up.
- **Numbers discipline**: this episode deliberately quotes no benchmark
  figures. If refinement adds any, each must carry the repo's framing
  on screen: dated snapshot (2026-07-04), machine-specific, losses
  recorded as plainly as wins (`docs/BENCHMARK.md`). "About thirty lines"
  in the VO refers to the README code block `README.md:24-53` (30 lines
  including comments and blanks) — keep the on-screen crop matching.
- The Qwen3 chat answer on camera is whatever the model actually prints —
  do not script or fake the output.
