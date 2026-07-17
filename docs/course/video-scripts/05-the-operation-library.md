# Video 05 — The operation library (3:00)

*Series: Forging Deep Learning in Zig · Source: ../05-the-operation-library.md*

## Logline
Fucina's op library is eager and local: every call validates once at a tiny
facade and dispatches to small unchecked kernels that never allocate. We
trace one op from facade to kernel in seconds, then contract by axis name —
deep learning's vocabulary as a library of functions you can read.

## Takeaways
1. **Eager and local.** An op call runs the kernel and returns an owned
   tensor immediately — no graph object, no session, no compile step, no
   lazy evaluation. The whole path is: validate once at the facade, then
   small unchecked kernels.
2. **Kernels never allocate.** Every op result comes from the context's
   buffer pool, so eager chains recycle a couple of warm buffers instead of
   churning an allocator.
3. **The vocabulary is readable.** Softmax, norms, losses, topk — a library
   of functions, with contraction chosen by axis name (`dot(..., .k)`), and
   the chapter's snippets machine-verified in CI.

## Script

### [0:00–0:20] No graph anywhere
**VO:** Here's a claim you can check in a debugger: when Fucina runs a
neural network, there is no graph object, no session, no compile step, no
lazy evaluation. You call an op — the kernel runs, a finished tensor comes
back. What you read is what runs. So today, we read it.
**Visual:** Diagram: two pipelines side by side — "capture graph → compile →
execute" grayed out on the left, "call → kernel → owned tensor" highlighted
on the right. Then the quote card: "an op call runs the kernel and returns
an owned result immediately" — docs/REFERENCE.md §6.
**Overlay:** "PyTorch is eager too — Fucina drops everything *around* the
eager core: no global device, no graph-capture modes."

### [0:20–1:00] One op, facade to kernel, in seconds
**VO:** So let's trace add — the whole path, three layers. Layer one, the
facade: one line. The result's axis names are computed at compile time; the
body just dispatches. Layer two, the lowering: validate both operands once,
build zero-copy broadcast views — a broadcast is a zero-stride view, never a
copy — then dispatch to rank-specialized kernels. Below this line, nothing
re-checks anything. Layer three: kernels classify layout exactly once —
contiguous, scalar, tail-broadcast, or arbitrary — and pick a fast path.
That is the entire architecture: validate once at the facade, then small
unchecked kernels.
**Visual:** Three code shots in descending sequence, with a small "depth
gauge" graphic on the left edge (facade → lowering → kernel) tracking the
descent: (1) `src/ag/tensor.zig:1241–1243` (`add`, the whole method);
(2) `src/tagged.zig:53–81` (`pointwise` — hold on the two `validateTensorRank`
lines, then the two `broadcastTensorTo` view lines, then the `switch`);
(3) `src/exec.zig:48–53` (`LayoutClass` enum).
**Overlay:** On shot 2: "validate once — nothing below re-checks". On shot 3:
"facade ≈ 100+ methods, raw surface ≈ 300 pub fns (counts as of the
chapter's writing; API young and unstable, no semver)".

### [1:00–1:35] Kernels never allocate
**VO:** One discipline holds across the whole kernel tier: kernels never
allocate. Every op result comes from the context's buffer pool — episode
three's recycling pool. Release an intermediate, and the next same-shaped
result gets the very same address back, warm in cache. And here is scale —
the library's whole skeleton in one tiny method: run the kernel, errdefer
the value, hand off to finishOp. Every differentiable op ends in one of two
shared tails, so the contract is implemented once, not re-derived per op.
**Visual:** Code shot: `src/ag/tensor.zig:1257–1261` (`scale`, entire
method). Then a small loop diagram: tensor `deinit` → buffer returns to the
pool's free list → next same-shaped op receives the same address.
**Overlay:** "kernels never allocate — docs/REFERENCE.md §6.5" · "same
address back: pinned by the pointer-equality recycling test (episode 03)".

### [1:35–2:10] Contraction by axis name
**VO:** Now watch contraction. dot doesn't ask you to arrange operands into
blessed layouts — you name the axis to contract. In this test, both tensors
share a `.b` axis, so it becomes a batch dimension automatically. That's
batched matrix multiply without remembering a separate b-m-m entry point —
and without a single transpose call anywhere. Orientation is the lowering's
problem, not yours. And this snippet is machine-verified: it runs in CI,
straight from the reference.
**Visual:** First a one-line shot of the `dot` signature,
`src/ag/tensor.zig:3747`. Then the test "dot with a shared batch tag lowers
to bmm" from the chapter, `docs/course/05-the-operation-library.md:508–521`,
highlighting `.k` in the `dot` call and the shared `.b` tag. Then a terminal
shot: `zig build snippet-check` (run in the repo root; show the tail of the
run completing successfully).
**Overlay:** "contract tag: named by you · batch tags: shared · free tags:
private — result = batch ++ left-free ++ right-free".

### [2:10–2:45] The vocabulary is a library
**VO:** And that's the pattern for the entire vocabulary of deep learning.
Twenty-seven activations in one closed enum — including three named GELU
variants, because matching a reference model's exact approximation is the
difference between bitwise parity and mysterious drift. Softmax's option
struct packs a decade of attention engineering: scale, additive masks,
ALiBi, sinks, fused causal. Cross-entropy on uniform logits returns log of
K — the entropy of pure ignorance. All of it eager, all of it local, all of
it readable.
**Visual:** Quick cuts of three chapter excerpts from
`docs/course/05-the-operation-library.md`: the activation table (lines
375–384, hold on the three gelu rows), the softmax options list (lines
618–629), and the crossEntropy ln(K) test (lines 744–759).
**Overlay:** "27 UnaryOp members — count as of the chapter's writing" · on
the loss test: "ln(4) ≈ 1.386, matched to 1e-6 in the test".

### [2:45–3:00] Close and teaser
**VO:** The chapter walks the whole vocabulary — masks, reductions, gather
and scatter, determinism as a contract. Next time: those small unchecked
kernels have a job to do. Going fast on CPUs.
**Visual:** End card: chapter link (docs/course/05-the-operation-library.md)
over a slow zoom-out of the three-layer depth-gauge diagram from segment 2.
**Overlay:** "Next: 06 — Going fast on CPUs".

## Asset list
- **Code shots (record from the repo at current main):**
  - `src/ag/tensor.zig:1241–1243` — `add`, the entire facade method.
  - `src/tagged.zig:53–81` — `pointwise`, the entire lowering function.
  - `src/exec.zig:48–53` — the `LayoutClass` enum.
  - `src/ag/tensor.zig:1257–1261` — `scale`, the entire method.
  - `src/ag/tensor.zig:3747` — the `dot` signature (one line, wraps).
- **Chapter excerpts (render from `docs/course/05-the-operation-library.md`):**
  - Lines 508–521 — machine-verified test "dot with a shared batch tag
    lowers to bmm".
  - Lines 375–384 — the activation-zoo table.
  - Lines 618–629 — the softmax options list.
  - Lines 744–759 — machine-verified test "crossEntropy on uniform logits
    is ln(K)".
- **Terminal recording:** `zig build snippet-check` from the repo root (the
  in-tree gate that extracts and runs every runnable docs/REFERENCE.md
  snippet against the real modules). Show the tail: the step completing
  without failures. No model downloads needed for this episode.
- **Diagrams (3):**
  1. Eager-vs-graph pipelines: "capture → compile → execute" grayed vs
     "call → kernel → owned tensor" highlighted.
  2. Depth gauge: three stacked layers labeled facade (`src/ag/tensor.zig`)
     → lowering (`src/tagged.zig`) → kernels (`src/exec.zig` and leaves),
     with a marker descending as the trace proceeds.
  3. Pool recycling loop: deinit → free list → same-shaped successor gets
     the same address.

## Production notes
- **Tone:** warm and concrete; the episode's energy is "look how short the
  path is" — let the three code shots breathe rather than rushing more
  material in. No hype; the series never claims state of the art.
- **Caveats that MUST stay attached:** the PyTorch-is-eager-too overlay in
  segment 1 (the eager design is a *scope* choice, not an invention claim);
  "counts as of the chapter's writing" on both the 27-member enum and the
  100+/~300 surface sizes (the API is explicitly young and unstable, no
  semver); the 1e-6 tolerance on the ln(K) overlay (the math is exact, the
  test asserts to 1e-6). No benchmark numbers appear in this episode — do
  not add any.
- **Pronunciation:** `finishOp` as "finish-op"; `.b` as "dot-b"; `bmm` as
  "b-m-m"; ln(K) in VO is spoken "log of K".
- **If the cut runs long, trim in this order:** (1) the `dot` signature
  beat at the top of segment 4 (keep the test); (2) the third chapter
  excerpt in segment 5 (the ln(K) test) and its VO sentence; (3) the quote
  card in segment 1 (keep the diagram and the PyTorch overlay).
- **Must not change:** the trace order facade → lowering → kernel in
  segment 2 (it *is* the core idea); the phrase "validate once at the
  facade, then small unchecked kernels"; the claim "kernels never allocate"
  with its §6.5 citation; the machine-verified framing of the dot test
  (that honesty beat is deliberate).
- `zig build snippet-check` runs the full snippet suite and may take a
  while; record once, cut to the final passing lines. Do not fake the
  output.
