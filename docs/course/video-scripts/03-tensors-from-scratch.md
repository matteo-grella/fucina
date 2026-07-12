# Video 03 — Tensors from scratch (3:00)

*Series: Forging Deep Learning in Zig · Source: ../03-tensors-from-scratch.md*

## Logline

A tensor is flat memory plus an interpretation — dtype, shape, strides — so
transpose, broadcast, and slicing are stride bookkeeping with zero copies.
The showcase: Fucina's buffer pool, where `deinit` recycles storage so the
next same-sized op gets the *same pointer* back, warm in cache — a design the
memory-model document adjudicates as "strictly better" than an arena.

## Takeaways

1. `address = offset + Σ index·stride` is the whole theory: transpose,
   broadcast, and slicing are metadata edits over one shared flat buffer —
   no data moves.
2. In Fucina, `deinit` is not cleanup ceremony; it drives a buffer pool that
   hands the same address back to the next same-sized op (pointer-equality
   asserted in a machine-verified REFERENCE.md test), keeping cache lines warm.
3. The pool was weighed against an arena in a dated design document and won
   on measured grounds: holding buffers arena-style (a held exec scope) hit
   32 distinct buffers on a 32-op 1 MiB chain; the pool's deinit-as-you-go
   discipline, 2.

## Script

### [0:00–0:15] Six floats, two readings

**VO:** Six floats, sitting in a row. Read them as two rows of three — that's
a matrix. Now swap four integers — it's the transpose. Nothing moved. That
trick is the entire theory of tensors.

**Visual:** Diagram (animated, drawn from chapter §3.1): a flat buffer
`[1 2 3 4 5 6]` with an "interpretation" card beside it — `shape {2,3},
strides {3,1}` rendering as a 2×3 grid; the card's numbers swap to
`shape {3,2}, strides {1,3}` and the grid re-renders as 3×2 while the buffer
visibly stays untouched.

**Overlay:** `shape {2,3} · strides {3,1}` → `shape {3,2} · strides {1,3}` —
"same six floats"

### [0:15–0:50] The formula

**VO:** A tensor is flat memory plus an interpretation: a dtype naming what
the bytes mean, a shape, and strides. The element at i, j lives at offset
plus i times stride zero, plus j times stride one. That formula is the whole
theory. NumPy works this way. PyTorch works this way. Fucina works this way.
And in Fucina a raw tensor is exactly four fields — buffer pointer, shape,
strides, offset — stored inline, so creating a view allocates nothing.

**Visual:** The address formula appears over the diagram
(`address = offset + i·strides[0] + j·strides[1] + …`, chapter §3.1); then a
code shot of the real struct: `src/tensor.zig:94-101` (fields `buffer`,
`shape`, `strides`, `offset`), highlighting the four fields.

**Overlay:** "address = offset + Σ index·stride" · then: "4 fields — no
allocation per view (docs/REFERENCE.md §8.5)"

### [0:50–1:20] Views for free — and the bill

**VO:** So the magic becomes bookkeeping. Transpose: swap two shape entries
and the same two stride entries. Broadcast: give an axis stride zero, and
every index along it reads the same memory — NumPy's celebrated broadcasting
rules turn out to be fifteen lines. Slicing: bump the offset, shrink the
shape. Zero copies, every time. But now two tensors share one buffer, and
both have a deinit. Fucina's answer is an atomic refcount: whoever returns
the last reference destroys the storage.

**Visual:** Quick code shot of the real broadcast loop,
`src/tensor.zig:464-483`, highlighting the three stride cases (new leading
axis → 0, matching axis → keep, size-1 axis → 0). On "who frees", cut to
`release` in `src/storage.zig:117-129`, highlighting `fetchSub` and
`if (old == 1)`.

**Overlay:** "stride 0 = broadcast — no data duplicated" · then: "fetchSub:
exactly one thread sees old == 1"

### [1:20–2:00] The pool: same pointer back

**VO:** Except it doesn't have to destroy it. The destructor is pluggable,
and Fucina plugs in a buffer pool. Here's the machine-verified test from the
reference manual. Run an op. Grab the result's data pointer. Deinit it. Run
the same-sized op again — and the pool hands back the exact same address.
Pointer equality, asserted. That's more than saved allocator work: the buffer
you just wrote is still sitting in L1 or L2 when the next op writes its
output to the very same lines.

**Visual:** Code shot of the test `"deinit recycles transient buffers through
the pool"` from `docs/REFERENCE.md:4500-4516` (§6.2). Step-highlight in sync
with the VO: `const first_ptr = first.dataConst().ptr;` → `first.deinit();`
→ `var second = try ctx.add(&a, &a);` → the final
`expectEqual(first_ptr, second.dataConst().ptr)`.

**Overlay:** "same size → same address — asserted, machine-verified
(REFERENCE.md §6.2)"

### [2:00–2:35] LIFO warmth, arena verdict

**VO:** The source says the quiet part out loud. Within a size class, the
pool hands back the most-recently-released buffer first — LIFO — because,
quoting the comment: its lines are the likeliest to still be cache-resident.
And why a pool instead of an arena? There's a design document adjudicating
exactly that. Its verdict: the buffer pool already is an arena — quote — but
a strictly better one. Holding buffers arena-style measured thirty-two
distinct buffers on a thirty-two-op chain. The pool's deinit-as-you-go
discipline: two.

**Visual:** Code shot of the in-source comment,
`src/exec/buffer_pool.zig:185-188` ("Insert BEFORE equal-size entries …
its lines are the likeliest to still be cache-resident"), the quoted phrase
highlighted. Then a doc shot of `docs/MEMORY-MODEL.md:10-15` with
"but a strictly better one" highlighted, followed by
`docs/MEMORY-MODEL.md:202-211` with "measured 2 vs 32 distinct buffers on a
32-op 1 MiB chain" highlighted.

**Overlay:** On the measurement: "2 vs 32 distinct buffers — 32-op × 1 MiB
chain; deterministic buffer count pinned by a named test
(docs/MEMORY-MODEL.md, adjudicated 2026-06-10)"

### [2:35–3:00] Deinit is the driver

**VO:** So deinit here isn't cleanup ceremony — it's the recycling driver.
Every defer you write feeds warm memory to the next operation. One honest
note: this raw layer is deliberately not public API — re-exporting it is a
compile error, enforced in source. What the supported surface adds isn't
machinery. It's meaning: axes with names, checked at compile time. That's
the next video.

**Visual:** Animated loop diagram (from chapter §3.12's chain):
`tensor.deinit()` → `buffer.release()` → refcount 0 → release hook → free
list → "next op, warm lines". On "compile error", a brief code shot of the
comptime guard `src/fucina.zig:33-42`. End card: series title + "Next:
Axes with names".

**Overlay:** "deinit → release → hook → free list → next op" · end card:
"04 — Axes with names: `fucina.Tensor(.{ .batch, .in })`"

## Asset list

- **Diagram 1** (0:00): flat buffer `[1 2 3 4 5 6]` + interpretation card;
  animate the shape/stride swap re-rendering 2×3 → 3×2. Source: chapter §3.1
  ASCII figure.
- **Code shots** (all real repo files, ranges verified 2026-07-12):
  - `src/tensor.zig:94-101` — the four-field raw tensor struct.
  - `src/tensor.zig:464-483` — the broadcast stride loop.
  - `src/storage.zig:117-129` — `release` with `fetchSub`.
  - `docs/REFERENCE.md:4500-4516` — the pointer-equality pool test (§6.2).
  - `src/exec/buffer_pool.zig:185-188` — the LIFO cache-warmth comment.
  - `docs/MEMORY-MODEL.md:10-15` and `docs/MEMORY-MODEL.md:202-211` — the
    arena verdict quote and the 2-vs-32 measurement.
  - `src/fucina.zig:33-42` — the comptime anti-export guard (brief).
- **Diagram 2** (2:35): the recycling loop
  deinit → release → hook → free list → next op.
- **End card**: series branding + teaser "04 — Axes with names".
- No terminal recordings and no model downloads are required for this episode.

## Production notes

- **Tone**: warm, concrete, unhurried; the hook and the pointer-equality beat
  are the two moments to land — give the final `expectEqual` highlight a
  visible beat before the VO moves on.
- **Quotes are quotes**: "its lines are the likeliest to still be
  cache-resident" (src/exec/buffer_pool.zig:185-188) and "but a strictly
  better one" (docs/MEMORY-MODEL.md) must appear as attributed source/doc
  text on screen, not as narrator claims.
- **Caveat that MUST stay with the number**: 2-vs-32 is a deterministic
  *buffer count* on a 32-op 1 MiB chain, pinned by the "exec scope holds
  buffers until close" test in `src/ag/tensor_tests.zig` and recorded in
  docs/MEMORY-MODEL.md (adjudicated 2026-06-10). It is not a wall-clock
  benchmark; do not restyle it as a speedup. The VO's "holding buffers
  arena-style" is the document's own framing — the measurement compares a
  held exec scope (evaluated there as "a deinit-eliminating arena") against
  deinit-ASAP through the pool; keep that pairing intact.
- **"Fifteen lines"** refers to the chapter's characterization of the
  broadcast loop at src/tensor.zig:464-483; keep the code shot attached to
  that phrase.
- **Optional terminal beat (settled)**: chapter 03 gives no runnable command
  for the pool test, so the §6.2 snippet stays a code shot. If production
  wants a live run anyway, `zig build snippet-check` is real and in-tree
  (build.zig:729; docs/REFERENCE.md §2.7 and the §4 target table; it is a CI
  step) — it extracts and runs *every* runnable REFERENCE.md snippet, this
  test included. If shown, frame it as the gate that machine-verifies the
  snippet, not as running this one test in isolation.
- **Honesty guardrails**: keep the "raw layer is deliberately not public API"
  note in the close (chapter §3.13 / REFERENCE.md §8 instability warning);
  no state-of-the-art or production-readiness claims anywhere.
- **If the cut runs long, trim in this order**: (1) the `src/storage.zig`
  release shot in segment 3 (keep the VO line, show it over the broadcast
  shot); (2) the `src/fucina.zig` guard shot in segment 6 (overlay text can
  carry it); (3) tighten Diagram 1's swap animation. Do not trim the
  pointer-equality test, the LIFO comment, or the arena verdict — they are
  the episode's showcase.
- **Line-range drift**: code/doc line ranges were verified against the tree
  on 2026-07-12; re-check before recording if the tree has moved.
