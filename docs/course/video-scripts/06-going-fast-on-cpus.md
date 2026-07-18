# Video 06 — Going fast on CPUs (3:00)

*Series: Forging Deep Learning in Zig · Source: ../06-going-fast-on-cpus.md*

## Logline

Matrix multiplication is where essentially all of a network's wall-clock time
goes — GEMM when many tokens are in flight, GEMV when there's one — and BLAS
has been the standard interface for exactly this operation since the Fortran
era. The video shows Zig's `@Vector` turning one source kernel into NEON and
AVX2 machine code, and Fucina's dispatch seam where CBLAS, Metal, and CUDA are
all just providers that may decline — with a benchmark record honest enough to
keep its losses on screen.

## Takeaways

1. One parameter — `m`, the tokens in flight — splits deep learning's one
   dominant op into two worlds: GEMM (prefill/training, compute-bound) and
   GEMV (decode, memory-bandwidth-bound).
2. `@Vector` is SIMD as a language feature: one portable kernel body compiles
   to NEON on an M1 and AVX2 on x86, specialized at compile time, unused ISA
   arms not in the binary.
3. Accelerators are providers, not architecture: CBLAS, Metal, and CUDA plug
   into the same GEMM dispatch seam and may always decline; pure Zig always
   answers — and every performance claim carries its machine, date, and its
   recorded losses.

## Script

### [0:00–0:28] Where the time goes

**VO:** Run any transformer under a profiler and the picture is lopsided:
matrix multiplication dominates, usually by an order of magnitude over
everything else combined. Linear layers are matmuls. Attention scores are a
matmul. The feed-forward block is two or three more. The vocabulary
projection is one enormous matmul. So the recipe for a fast CPU runtime is
unromantic: make matmul fast, make everything else not slow, and measure
honestly.

**Visual:** Schematic horizontal bar diagram (illustrative, not a recorded
profile): one long bar labeled "matmul" dwarfing a sliver labeled "everything
else", per §6.1's "order of magnitude" description. As the VO lists layers,
small tags pop onto the matmul bar: *linear · attention scores · attn·V ·
FFN ×2–3 · vocab projection*.

**Overlay:** "matmul dominates — usually by an order of magnitude (ch. 6, §6.1)" ·
small caption on the diagram: "schematic".

### [0:28–0:59] One letter, two workloads

**VO:** One parameter flips the physics: m, the tokens in flight. Prefill and
training keep m large — that's GEMM, matrix times matrix; every weight is
reused m times, so the work is compute-bound. Decode generates one token at a
time, m equals one — the GEMM degenerates to a GEMV, matrix times vector, and
every weight byte is read for a single multiply-add. Memory-bandwidth-bound.
Same operation family, opposite economics, and that split drives dispatch
decisions throughout the library.

**Visual:** Animated diagram of `C = A·B` with labeled shapes `m×k · k×n →
m×n`. First state: A is tall ("prefill / training: m large"), caption
"GEMM — each weight reused m times → compute-bound". Then A collapses to a
single row ("decode: m = 1"), caption "GEMV — every weight byte read for one
multiply-add → memory-bandwidth-bound". Source: the ML note in §6.1.

**Overlay:** "m large → GEMM · compute-bound" / "m = 1 → GEMV ·
bandwidth-bound".

### [0:59–1:33] One kernel, two instruction sets (showcase)

**VO:** Now the showcase. Fucina's entire vector-width policy is two lines:
ask the target how wide its SIMD is, make that a first-class vector type. On
@Vector operands, plus is a SIMD add. Here is the real vecAdd: a
four-times-unrolled vector body, a one-vector body, a scalar tail. On an M1
this compiles to NEON fadd.4s. On an AVX2 machine, vaddps on ymm registers.
Same body on both. One source tree, specialized at compile time — the
unused ISA arms aren't even in the binary.

**Visual:** Code shot 1: `src/backend/vector/common.zig:24–25` (the two-line
vector-width policy), held while the VO explains it. Code shot 2:
`src/backend/vector/primitives.zig:52–74` (the production `vecAdd`), with the
three tiers highlighted in sequence: 4×-unrolled body, 1× vector body, scalar
tail. Then a split-screen graphic: the same source centered, left panel
"Apple M1 → NEON `fadd.4s` (4 lanes)", right panel "x86 AVX2 → `vaddps` on
`ymm` (8 lanes)" — per §6.5; render as annotation, do not attempt live
disassembly.

**Overlay:** "`std.simd.suggestVectorLength(f32)` → 4 on NEON · 8 on AVX2" ·
"one source, per-target machine code".

### [1:33–2:07] BLAS heritage and the provider seam

**VO:** So where does BLAS fit? It was born in the Fortran era as the standard
interface for exactly this operation family — GEMM is its Level-3 flagship,
and the reference implementation is Fortran to this day. Decades of tuning
live behind it; refusing them would be vanity. So the dispatch is a
fall-through chain: GPU, if built in, may decline; BLAS, if built in, may
decline; pure Zig always answers. CBLAS, Metal, CUDA — all plug into the same
GEMM seam. GPU as provider, never the architecture.

**Visual:** Brief history card while the VO covers heritage: "BLAS — Fortran
era · Level 1 vector–vector / Level 2 matrix–vector (GEMV) / Level 3
matrix–matrix (GEMM) · SGEMM/DGEMM naming = Fortran heritage" (stated as
common knowledge, per §6.6). Then code shot:
`src/backend/native.zig:171–193` — the dispatch precedence — with three
sequential highlights synced to the VO: the `use_gpu` block ("may decline"),
the `use_blas` block ("may decline"), the final
`vector.matmul2DIntoUncheckedWithConfig` line ("always answers").

**Overlay:** "`-Dgpu=metal|cuda` · `-Dblas=accelerate|openblas|mkl|…` · pure
Zig always answers" · "`comptime` guards: a `-Dblas=none -Dgpu=none` build
contains *neither* upper tier — not disabled, absent".

### [2:07–2:39] Honest numbers — a win and a loss

**VO:** And every number wears its protocol. Paired benchmarks run in both
process orders; rows noisier than eight percent aren't counted as results;
medians, not best cases. The headline, from a dated snapshot on an M1 Max,
CPU-only on both sides: of 236 paired cells against llama.cpp, Fucina is
faster in 221. And a loss stays on the board: Qwen3.5 Q8_0 at prompt length
32 measures 0.86 times — recorded, caveated, kept until a better measurement
replaces it.

**Visual:** Two side-by-side cards rendered from `docs/BENCHMARK.md` /
README.md as quoted in §6.9. WIN card: "236 paired cells · faster in 221 · at
parity in 13 · dense prefill geomeans 1.18–1.81× per format". LOSS card:
"Qwen3.5-0.8B Q8_0 pp32 → 0.86×". Above both, three protocol bullets:
"both process orders · CV > 8% = NOISY, not counted · medians, not best
cases" (tools/bench_gate.py).

**Overlay:** Persistent caption under both cards: "Snapshot 2026-07-04 ·
Apple M1 Max · CPU-only both sides · 8 threads · llama.cpp with its
Accelerate BLAS backend (its default on this platform)". On the LOSS card:
"confined to pp32 — pp128 1.09×, pp512 1.17×, decode 1.37× all win;
diagnosed-but-open bimodality".

### [2:39–3:00] The discipline, and what's next

**VO:** That's the whole discipline: one portable source, a seam any
accelerator can plug into, and a record that keeps its losses. The chapter
walks the full GEMM staircase — loop order, register tiles, cache blocking, a
5.6-times win at large sizes. Next time: autograd — the graph hidden in the
values.

**Visual:** Staircase graphic, five steps rising left to right: "naive loops →
loop reorder → register tiling → cache-blocked packing → provider tiers
(BLAS/GPU)", per §6.6. The 5.6× lands on the fourth step with its caveat
attached. End card: series title, "Full chapter:
`docs/course/06-going-fast-on-cpus.md`", "Next: 07 — Autograd: the graph
hidden in the values".

**Overlay:** "5.6× — 2048³, 109→608 GFLOP/s · M1 Max, `-Dblas=none` build ·
docs/BENCHMARK.md" · end card: "full chapter in `docs/course/`" · "Next:
Autograd — the graph hidden in the values".

## Asset list

**Code shots (repo files, exact ranges):**
- `src/backend/vector/common.zig:24–25` — the two-line vector-width policy.
- `src/backend/vector/primitives.zig:52–74` — the production `vecAdd`
  three-tier loop.
- `src/backend/native.zig:171–193` — GPU → BLAS → vector dispatch precedence.

**Diagrams to render (one sentence each):**
- Schematic profiler bar: "matmul" vs "everything else", order-of-magnitude
  gap, labeled "schematic" (§6.1 prose, not a recorded profile).
- GEMM→GEMV morph: `C = A·B` with `m×k · k×n` labels; A collapses from tall
  matrix to single row as m→1, with compute-bound/bandwidth-bound captions
  (§6.1 ML note).
- Split-screen "one source, two ISAs": `vecAdd` center, NEON `fadd.4s` (4
  lanes) left, AVX2 `vaddps` on `ymm` (8 lanes) right — annotation only,
  claims from §6.5.
- BLAS history card: Fortran era, Levels 1/2/3, GEMM = Level-3 flagship,
  SGEMM/DGEMM naming (§6.6, common knowledge).
- Benchmark WIN/LOSS cards + protocol bullets, all text quoted from §6.9
  (sources: `docs/BENCHMARK.md`, README.md, `tools/bench_gate.py`).
- GEMM staircase with 5.6× caveat on step four (§6.6, `docs/BENCHMARK.md`).
- End card with "Full chapter: `docs/course/06-going-fast-on-cpus.md`" and
  next-episode teaser.

**Terminal (optional, type-on only — do not execute on camera):** build-flag
montage `zig build -Dblas=accelerate` / `zig build -Dgpu=metal` /
`zig build -Dgpu=cuda` over the seam segment; flags per the §6.2 build-options
table. If an executed shot is wanted instead, `zig build test` (runs the
cross-backend parity suite regardless of `-Dbackend`, §6.4) is the safe
choice.

**External downloads:** none — no model weights needed; all numbers are
quoted from the repo's benchmark record, not re-measured.

## Production notes

- **Tone:** confident and concrete, zero hype. The loss segment is delivered
  with the same energy as the win — that contrast *is* the point of the
  episode's closing third. Never frame 0.86× apologetically; frame it as the
  protocol working.
- **Caveats are load-bearing and MUST NOT be cut:** (a) the benchmark
  conditions caption (snapshot 2026-07-04, M1 Max, CPU-only both sides,
  8 threads, llama.cpp on its Accelerate backend) must be on screen whenever
  221/236 or 0.86× is; (b) the 5.6× must carry "2048³ · M1 Max ·
  `-Dblas=none` build"; (c) the profiler bar must stay labeled "schematic".
- **Do not** attempt live disassembly for the NEON/AVX2 beat; the
  `fadd.4s`/`vaddps` claims are quoted from the chapter (§6.5) and rendered
  as annotations. Do not execute the build-flag montage; it is type-on only.
- **If the cut runs long, trim in this order:** the BLAS history card's
  on-screen dwell (keep the VO), then the staircase graphic's step-by-step
  animation (land it as a single frame), then the terminal type-on montage
  (fully optional). Never trim the LOSS card or any caveat overlay.
- **Numbers appearing in the video and their sources:** order-of-magnitude
  matmul dominance (§6.1, qualitative); 4/8 lanes (§6.5); 236/221/13 cells and
  1.18–1.81× geomeans (README.md + `docs/BENCHMARK.md` via §6.9); 0.86× pp32
  with pp128/pp512/decode context (§6.9); 5.6× = 109→608 GFLOP/s at 2048³
  (`docs/BENCHMARK.md` via §6.6); CV > 8% NOISY rule (`tools/bench_gate.py`
  via §6.9). Nothing else may be quantified.
- The next-episode teaser line ("Autograd — the graph hidden in the values")
  matches Video 07's title and must survive edits.
