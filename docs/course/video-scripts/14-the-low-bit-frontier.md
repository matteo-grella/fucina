# Video 14 — The low-bit frontier (3:00)

*Series: Forging Deep Learning in Zig · Source: ../14-the-low-bit-frontier.md*

## Logline
Weights that take only three values — −1, 0, +1 — don't just shrink storage
to ~2 bits: multiplication disappears from the matmul kernel entirely. We
show the TQ2_0 trick (store `w+1`, subtract once per block), the measured
speedups it buys on two CPUs, the straight-through estimator that trains
through a zero-gradient quantizer ("wrong, and it works"), and PTQTP's
honest scoreboard: perplexity parity at 1.9× BF16 decode — with the
collapses and the baseline that flips the verdict stated out loud.

## Takeaways
1. **Ternary changes the kind of kernel, not just its size**: TQ2_0 stores
   codes `w+1 ∈ {0,1,2}` in 2-bit crumbs, and
   `dot(w,a) = Σ(w+1)·a − Σa` turns the matmul into unsigned int8 dots
   plus one subtraction per block — zero weight multiplications, with `Σa`
   already precomputed in Q8_K's `bsums`.
2. **The straight-through estimator trains what a gradient cannot see**:
   forward with the quantized weight, backward pretending the quantizer is
   the identity — mathematically indefensible, empirically what the whole
   QAT literature runs on; the latent float weight is a vote counter.
3. **A frontier result is only as good as its baselines**: PTQTP K=3 from a
   BF16 1.7B source reaches perplexity parity (18.43 vs 18.57) at 1.9× the
   BF16 original's decode speed on M1 Max; K=2 collapses ×17 at that
   scale; and against Q4_K_M the same K=3 is ~0.75× on ARM while the
   documented per-plane ratios flip the verdict on x86-VNNI.

## Script

### [0:00–0:25] Three values
**VO:** Push quantization to its edge: a weight that can only be minus one,
zero, or plus one. Two things happen, and they are different in kind.
Storage drops to about two bits per weight. And — the deeper one —
multiplication disappears from the inner loop. Multiplying by minus one is
negation. By zero, skipping. By plus one, copying. The ternary kernel
contains no weight multiplications at all.
**Visual:** Diagram: a multiply unit labeled `×` between a weight and an
activation dissolves into three arrows — `×(−1) → negate`, `×0 → skip`,
`×(+1) → copy` — then the weight snaps to one of three dots on a
{−1, 0, +1} number line.
**Overlay:** `{−1, 0, +1} · ~2 bits/weight · no multiplications`

### [0:25–0:55] The kernel with no multiplications
**VO:** The container is ggml's TQ2_0: sixty-six bytes for two hundred
fifty-six weights. It doesn't store the trit; it stores the code — w plus
one — which is zero, one, or two. Unsigned. So the dot product becomes: sum
of codes times activations, minus the sum of the activations. The first
term is exactly what int8 dot instructions want. The second is already
precomputed in the activation blocks. One subtraction per block buys back
the sign.
**Visual:** Code shot: `BlockTQ2_0` at `src/dtype.zig:213-216` with the
size assert at `src/dtype.zig:253` (`== 66`) beneath it; then the chapter's
compile-checked identity test at
`docs/course/14-the-low-bit-frontier.md:58-73`; end on the production inner
loop at `src/backend/quant/ternary.zig:344-354` with
`const isum = dots[ci] - bsum;` highlighted.
**Overlay:** `dot(w,a) = Σ(w+1)·a − Σa` · `Σa is free: Q8_K bsums already
carry it`

### [0:55–1:25] What 2.06 bits buys
**VO:** What does that buy? On an M1 Max, single thread, the hot kernel
runs a 4096-square matrix-vector product in 238 microseconds — 4.25 times
the scalar reference, and about 2.1 times the tuned 4-bit kernel. That is
just the byte ratio: two bits against four and a half. On a Raptor Lake x86
core it is 5.1 times, and 4.8 times Q4_K — VNNI eats thirty-two bytes per
instruction. Dated, machine-named snapshots — measured, not asserted.
**Visual:** Rendered table: the m=1 rows of the two measured tables from
`docs/TERNARY.md:63-68` (M1 Max) and `docs/TERNARY.md:82-87` (Raptor
Lake) side by side — columns shape/cold/hot/hot-cold/Q4_K. Optional B-roll
underneath: terminal recording of
`zig build bench-ternary -Doptimize=ReleaseFast` on the production
machine, clearly labeled as a live run.
**Overlay:** `single thread, ReleaseFast, 2026-07-07 — your hardware will
differ` · `ARM kernel ≈ 86% of the NEON ALU roofline (docs/PTQTP.md)`

### [1:25–2:00] Wrong, and it works
**VO:** But how do you train weights a gradient cannot see? The quantizer
is a step function; its derivative is zero almost everywhere.
Backpropagate honestly and nothing learns. The fix is the straight-through
estimator: forward with the quantized weight, backward pretending the
quantizer is the identity. It is mathematically indefensible — you use the
gradient of a function you never evaluated — and it works. The latent
float weight becomes a vote counter: small pressures accumulate until a
trit flips. Fucina ships it as one op, dotTernarySte, exactly the BitNet
recipe.
**Visual:** Code shot: the chapter's compile-checked STE test at
`docs/course/14-the-low-bit-frontier.md:214-233`, with the forward line
(`const y = quantizeTrit(w, d) * x;` — quantized) and the backward line
(`const dw = dy * x;` — as if it were `w * x`) highlighted in sequence;
close on the final assert (the trit flipped to −0.5).
**Overlay:** `forward: quantized · backward: pretend it's smooth` →
`"wrong", and it works — the recipe the QAT literature runs on`

### [2:00–2:40] The honest scoreboard
**VO:** Now the frontier result. PTQTP takes a model you already have — no
gradients, no data — and rewrites each matrix as a sum of ternary planes.
The honest scoreboard, on an M1 Max: two planes collapse — perplexity
times seventeen at 1.7B. Three planes from the BF16 original: 18.43
against a baseline of 18.57 — parity — at 1.9 times the BF16 decode
speed. Change the baseline to Q4_K_M, and three planes run at about 0.75
times on ARM — while on x86 VNNI, the documented per-plane ratios say
multi-plane wins. A frontier result stated with its baselines is worth ten
stated without.
**Visual:** Diagram (one beat): `W ≈ Σₖ diag(αₖ)·Tₖ` — a weight matrix
splitting into K stacked ternary planes, each a valid TQ2_0 tensor. Then
two rendered tables from the chapter's §14.7: the perplexity ladder
(`docs/PTQTP.md:255-259`) with the K=2 column tinted red and the K=3
column green, then the decode-speed table (`docs/PTQTP.md:271-277`) with
the K=3 row highlighted.
**Overlay:** `M1 Max, 2026-07 snapshot · ppl = teacher-forced NLL over 512
held-out tokens` · on the x86 line: `x86 figure = arithmetic from the
documented per-plane ratios, not a separate measurement`

### [2:40–3:00] Reading a frontier
**VO:** Everything here is labelled research frontier, on purpose. The
identities and the kernels are stable — cross-ISA, bitwise, pinned by
tests. The speed ratios are dated snapshots on two machines, and the
accuracy story is genuinely unsettled. Reading a frontier honestly is a
skill. Next time: training language models on your CPU.
**Visual:** Split card: "Stable" (the mul-free identity, three training
paradigms, one wire format) vs "Provisional" (every speed ratio, the
ARM/x86 economics flip, trained-from-scratch ternary at scale — an open
question that isn't Fucina's to answer); end card with "Full chapter:
`docs/course/14-the-low-bit-frontier.md`".
**Overlay:** `stable: the code · provisional: the numbers` ·
`full chapter in docs/course/` · `Next: 15 — Training LLMs on your CPU`

## Asset list
- **Repo code shots**:
  - `src/dtype.zig:213-216` — `BlockTQ2_0` struct; pair with the
    `@sizeOf == 66` assert at `src/dtype.zig:253`.
  - `src/backend/quant/ternary.zig:344-354` — the tile inner loop
    (highlight `const isum = dots[ci] - bsum;`).
- **Chapter code excerpts** (compile-checked course code; source of truth
  is the chapter):
  - `docs/course/14-the-low-bit-frontier.md:58-73` — the mul-free identity
    test (`dot(w, a) = sum((w+1)*a) - sum(a)`).
  - `docs/course/14-the-low-bit-frontier.md:214-233` — the STE
    one-scalar training test.
- **Tables to render** (reproduce numbers exactly; keep dates and machine
  names in frame):
  - `docs/TERNARY.md:63-68` (M1 Max) and `docs/TERNARY.md:82-87` (Raptor
    Lake) — m=1 rows; if the dense-f32 column is shown, its Accelerate
    footnote (`docs/TERNARY.md:70-71`) must be visible.
  - `docs/PTQTP.md:255-259` — perplexity ladder (baseline / K=2 /
    K=2+down3+o3 / K=3).
  - `docs/PTQTP.md:271-277` — decode-speed table (t/s, vs baseline, ppl).
- **Terminal recordings** (both optional B-roll):
  - `zig build bench-ternary -Doptimize=ReleaseFast` — live kernel bench;
    label as live, keep the doc-snapshot overlay as the quoted source.
  - `zig build ptqtp-spirals -Doptimize=ReleaseFast` — the self-verifying
    PTQTP toy demo (single plane collapses, K=2 recovers 1.000 on the
    deployed int8 path); no model download needed.
- **Diagrams** (2): the "multiply dissolves" three-arrow diagram (0:00);
  the plane-decomposition split `W ≈ Σₖ diag(αₖ)·Tₖ` (2:00).
- **External downloads**: none required. All model-scale numbers
  (perplexity, decode t/s) are quoted from `docs/PTQTP.md` as dated
  design-record snapshots — do NOT attempt to reproduce them live; that
  would need Qwen3 0.6B/1.7B GGUFs (not in the repo) and an M1 Max.

## Production notes
- **Tone**: this is the series' "reading research honestly" episode — the
  scoreboard segment must feel like a researcher showing you the failures
  next to the win, not a victory lap. Let "perplexity times seventeen"
  land before the parity number arrives.
- **Caveats that MUST stay attached to numbers**:
  - Every kernel figure carries `single thread, ReleaseFast, 2026-07-07`
    and its machine name (M1 Max / i9-13950HX Raptor Lake); the VO's
    "dated, machine-named snapshots — measured, not asserted" must not be
    cut.
  - The "×17" collapse belongs to K=2 at 1.7B (330.96 vs 19.41 from the
    Q4_K_M-source line; the BF16-source line reads 184.45 vs 18.57) — show
    the table rather than re-deriving the ratio.
  - Parity is 18.43 vs 18.57, quoted exactly; "1.9×" is vs the model's own
    BF16 original on M1 Max, fused entry, interleaved runs.
  - The "~0.75× vs Q4_K_M" is ARM-only, and the x86-VNNI "multi-plane wins"
    claim is arithmetic from documented per-plane ratios (~2.1× ARM /
    ~4.8× x86-VNNI), not a separate measurement — the overlay saying so is
    load-bearing and must survive any trim.
  - No state-of-the-art claims, no production-readiness claims; the
    chapter's own label is "research frontier" and the video keeps it.
- **If the cut runs long, trim in this order**: (1) the optional terminal
  B-roll beats; (2) the Raptor Lake half of segment 3 (keep the M1 Max
  numbers and the "your hardware will differ" overlay); (3) the
  `ternary.zig` inner-loop cut in segment 2 (the course-code identity test
  carries the idea); (4) the NEON-roofline overlay.
- **The VO is at the 450-word contract cap** — do not add narration; any
  addition must be paid for by an equal cut (use the trim order above).
- **Pronunciation**: `dotTernarySte` is spoken as the identifier —
  "dot-ternary-S-T-E".
- **Do not change**: the quoted numbers and their dates; the
  "wrong, and it works" framing of STE (it is the chapter's point, not a
  disparagement — say it with a smile, not an apology); the closing
  stable-vs-provisional split; "A frontier result stated with its baselines
  is worth ten stated without" (chapter line, keep verbatim or drop
  whole); the BitNet-recipe attribution on `dotTernarySte`.
- The chapter has far more (ternary ES with undo-log restore, the exact
  mul-free f32 training path, two-encoder subtlety, the streaming
  quantizer that does a 30B MoE in 8.5 minutes at ~106 MiB); the end card
  sends viewers to the chapter rather than compressing it.
