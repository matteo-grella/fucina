# Video 07 — Autograd: the graph hidden in the values (3:00)

*Series: Forging Deep Learning in Zig · Source: ../07-autograd.md*

## Logline
Backward without a tape, a graph object, or an engine: each result carries a
pointer to the op that produced it, and `backward()` walks the pointers. We
build the whole scheme in ~60 lines of scalar Zig, hit the classic
double-counting bug, fix it with a pending counter — then find the identical
counter, atomic now, running Fucina's real engine, where the same forward
function trains under an exec scope and every gradient is checked against
finite differences.

## Takeaways
1. **The graph is implicit in the values** (the spaGO-inherited idea): no
   tape, no graph object, no central engine — each result points at the op
   that made it; backward discovers the topology by walking pointers.
2. **Shared nodes break naive recursion; a pending counter fixes it** —
   count consumer edges in a prepare pass, fire a node only when its counter
   drains. The ~60-line toy and `src/ag/core.zig` are recognizably the same
   code; Zig's rethink of spaGO's goroutines is an atomic counter draining
   onto a bounded pool.
3. **Same forward trains; gradients are verified, not trusted** — open an
   exec scope and inference-style code trains unchanged; `gradcheck` checks
   every VJP against a finite-difference oracle that can't share its bugs.

## Script

### [0:00–0:30] Where is the graph?
**VO:** Where does a deep learning framework keep its computation graph?
Usually in a tape, a graph object, or a trace — a registry that lives apart
from your values, with its own lifetime and its own rules. Fucina's answer
is: nowhere. Each result carries a pointer to the operation that produced
it. Backward discovers the topology by walking those pointers. Deinit the
tensors, and the graph is gone — because it never was anything but the
tensors.
**Visual:** Diagram: three small boxes labeled "tape", "graph object",
"trace" sitting beside a row of value nodes, then all three boxes dissolve,
leaving only the values with arrows pointing backward from each result to
the op inputs that produced it.
**Overlay:** `no tape · no engine · no graph object — the live tensors ARE
the graph`

### [0:30–0:55] Build it in sixty lines
**VO:** You can build the whole idea in about sixty lines of scalar Zig —
the chapter's compile-checked course code. A Value holds its data, a slot
for its gradient, and a tagged union recording which op made it, with
pointers to the operands. That field is the graph. For simple chains,
recursive backward just works.
**Visual:** Code shot: the `Value` struct with `data` / `grad` /
`rule: Rule` tagged union plus `add`/`mul`, from
`docs/course/07-autograd.md:124-142`, with the `rule` field and the
`[2]*Value` payload highlighted.
**Overlay:** `course code — compiles & passes under zig test (Zig 0.16.0)`

### [0:55–1:30] The bug, and the counter
**VO:** Then you share a node, and it breaks. Let s be x plus one, and y
equal s times s. Calculus says the gradient of x is six. The naive walk
delivers nine — the second visit through s re-sends the first contribution
on top of the second. Any reused weight, any residual connection builds this
diamond. The fix is to count: one prepare pass counts a pending contribution
per consumer edge; a node accumulates silently, and fires only when its
counter drains to zero. Complete, and exactly once.
**Visual:** Animated diamond diagram: `x → s → y` with two edges `s → y`;
the naive trace ticks `x.grad` to 3, then 9 (flagged red vs the correct 6);
then cut to the fixed `prepare`/`contribute` pair from
`docs/course/07-autograd.md:251-280`, followed by a terminal shot:
`zig test scalar_autograd.zig` passing both tests (file assembled from the
chapter — see asset list).
**Overlay:** `dy/dx = 2s = 6 — naive walk says 9` → then
`pending: count edges, fire at zero`

### [1:30–2:10] The real engine, and the spaGO lineage
**VO:** That counter is the real engine. GradState, in core dot zig, is the
toy in grown-up clothes: gradient accumulator, a type-erased VJP record, and
the pending counter — atomic now, because contributions arrive from a worker
pool. The design started in spaGO, in Go, where every node was a goroutine
blocking for its gradients. Zig has no goroutines, so readiness became data:
a counter draining to zero on a bounded pool, no blocked workers. As far as
I know, mainstream stacks route backward through a central engine. Here, the
live tensors are the graph.
**Visual:** Split screen: toy `Value` (chapter code) beside `GradState` at
`src/ag/core.zig:100-115`, matching fields connected by lines
(`grad`↔`grad`, `rule`↔`grad_fn`, `pending`↔`pending_grads`); then a brief
cut to the drain at `src/ag/core.zig:273-277`
(`fetchSub(1, .acq_rel)` … `return old == 1;`), then the Origins paragraph
at `README.md:274-286` with "(AFAIK) Mainstream stacks…" visibly on screen.
**Overlay:** `src/ag/core.zig — 733 lines incl. two in-file tests (wc -l,
as counted today)` · `(AFAIK)` kept on screen with the mainstream contrast

### [2:10–2:38] The same forward trains
**VO:** Now the payoff. This forward function, from the README's front page,
is written pure inference-style — defer, deinit, every intermediate freed as
consumed. Open an exec scope around it, and the same function trains,
unchanged: the scope adopts every intermediate, each deinit becomes a safe
no-op, and the whole graph stays alive until backward. There is no training
engine — every differentiable op ends in finishOp, and training is literally
one branch.
**Visual:** Code shot: the exec-scope snippet with its explanatory comment
at `README.md:43-53`; then `finishOp` at `src/ag/tensor.zig:6291-6308` with
the first line (`if (!wants_grad or !control.isGradEnabled()) return
finishNoGrad(...)`) highlighted.
**Overlay:** `"training and inference produce identical values" —
docs/REFERENCE.md`

### [2:38–3:00] Verified, not trusted
**VO:** And gradients are verified, not trusted. gradcheck nudges every
input element by epsilon and compares the measured slope against the
analytical backward — finite differences share no code with the VJPs, so
they can't share their bugs. Sixty lines to build it, one oracle to check
it. Next time: training — making the machine learn.
**Visual:** Code shot: the machine-verified `test "customVjp validated by
gradcheck"` snippet from `docs/course/07-autograd.md:1167-1185`, with the
`fucina.gradcheck(&ctx, squareLoss, .{&x}, .{})` call highlighted; end card
with chapter link `docs/course/07-autograd.md`.
**Overlay:** `|g_num − g_ana| ≤ abs_tol + rel_tol·|g_ana|` ·
`Next: 08 — Training`

## Asset list
- **Chapter code excerpts** (render as code shots, source of truth is the
  chapter, compile-checked course code):
  - `docs/course/07-autograd.md:124-142` — `Value` struct + `add`/`mul`.
  - `docs/course/07-autograd.md:251-280` — `prepare` + `contribute` (the
    pending-counter fix).
  - `docs/course/07-autograd.md:1167-1185` — `squareLoss` +
    `test "customVjp validated by gradcheck"`.
- **Repo code shots**:
  - `src/ag/core.zig:100-115` — `GradState` struct.
  - `src/ag/core.zig:273-277` — `finishGradContributionReady` drain.
  - `src/ag/tensor.zig:6291-6308` — `finishOp`.
  - `README.md:43-53` — exec-scope snippet with comment.
  - `README.md:274-286` — Origins paragraph (keep "(AFAIK)" visible).
- **Terminal recording**: assemble `scalar_autograd.zig` in a scratch
  directory by concatenating, verbatim, the chapter's fixed-engine block
  (`docs/course/07-autograd.md:230-294`) and the two passing tests
  (`docs/course/07-autograd.md:308-325`); run `zig test scalar_autograd.zig`
  with Zig 0.16.0 and capture the passing result (assembly re-verified:
  both tests pass). (Optional extra beat: a second file with the naive
  version also passes, re-verified — assemble it as
  `const std = @import("std");`, then the `Value` struct plus `add`/`mul`
  from lines 124-142 with the naive `backward`/`propagate` methods (lines
  159-181) inserted inside the struct body, then the deliberately-wrong
  test at 204-211 — the chapter keeps the failure mode on record.)
- **Diagrams** (2): the "three boxes dissolve" framework-comparison diagram
  (0:00); the animated diamond trace `x → s → y` showing 3 → 9 vs correct 6
  (0:55).
- **External downloads**: none. No model weights needed for this episode.

## Production notes
- **Tone**: warm and concrete; the diamond-bug trace is the emotional center
  of the first half — let the wrong "9" land before naming the fix. The
  exec-scope beat is the capability showcase — the point is "the SAME
  forward function", so keep the README snippet's comment block readable on
  screen.
- **Caveats that MUST stay attached**:
  - The **"(AFAIK)" hedge** stays on screen (and in VO as "as far as I
    know") wherever the mainstream-engine contrast is voiced — it is part of
    the README quote, not decoration.
  - "~60 lines" refers to the chapter's **course code**, not repo code; the
    overlay at 0:30 carries "compiles & passes under zig test (Zig 0.16.0)".
  - The "733 lines" overlay keeps its framing: "incl. two in-file tests,
    wc -l, as counted today".
  - No state-of-the-art or production-readiness claims anywhere.
- **If the cut runs long, trim in this order**: (1) the optional
  naive-version terminal beat; (2) shorten the Origins on-screen dwell to an
  overlay quote (the hedge must survive the trim); (3) the 733-lines
  overlay; (4) the `finishOp` cut at 2:10 (keep the REFERENCE.md quote
  overlay).
- **Do not change**: the 6-vs-9 trace numbers (they are the chapter's worked
  example); the "verified, not trusted" framing; the REFERENCE.md quote
  "training and inference produce identical values" (quote it exactly or
  drop it); the spaGO attribution (goroutine-per-node → atomic counter is
  the lineage story, not a footnote).
- The chapter has much more (VJP record anatomy, seeding rules, one-backward-
  per-graph, the lifetime rule, checkpointing); the end card sends viewers
  there rather than compressing it.
