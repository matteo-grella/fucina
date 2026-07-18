# Video 04 — Axes with names (3:00)

*Series: Forging Deep Learning in Zig · Source: ../04-axes-with-names.md*

## Logline

Axis names live in the tensor's *type*: `Tensor(.{ .batch, .in })` is a
different type from `Tensor(.{ .in, .batch })`, contraction happens by name,
and a misaligned contraction is a compile error — shown verbatim on screen, a
shape bug that never ran. Honestly bounded: names are compile-time, sizes are
runtime.

## Takeaways

1. `Tensor(spec)` is an ordinary Zig function returning a type; the axis
   names are part of that type, so reordered tags are a *different* type and
   `dot` contracts by name, with result tags computed while the compiler runs.
2. A contraction over an axis an operand doesn't have fails in `dot`'s
   *return type* — the wrong-shaped program is never compiled, let alone run.
3. The honest boundary: names and ranks are compile-time; *sizes* are runtime
   and fail with ordinary `error.ShapeMismatch`. Names catch the quiet bugs;
   sizes stay the loud kind.

## Script

### [0:00–0:22] The bug that doesn't crash

**VO:** Ask anyone who maintains model code what their most common bug is. It
isn't a wrong formula — it's a shape bug. And the nasty ones don't crash.
Multiply a transposed weight matrix the wrong way round and you often get a
plausible shape full of wrong numbers. The only symptom is a loss curve that
refuses to fall.

**Visual:** Diagram: two matrices multiplied the wrong way round produce a
plausibly-shaped output grid glowing red with wrong values, next to a
training-loss curve that stays flat.

**Overlay:** "The nasty shape bugs don't crash."

### [0:22–1:00] Names in the type

**VO:** Fucina's answer is the most Zig idea in the library: axis names live
in the tensor's type. An axis is a tag — batch, in, class — and Tensor of
batch-then-in is a different type from Tensor of in-then-batch. Pass one
where the other is expected: ordinary type error. Zig has no generics syntax;
Tensor is a plain function that takes the tags and returns a type. Read this
forward pass by its signature alone. Data enters as batch by in, and leaves
as batch by class. You can check the algorithm against the math by eye.

**Visual:** Code shot: `README.md:24–41` (the `Model` struct and `forward`),
first highlighting the field types `Tensor(.{ .h1, .in })` etc., then holding
on the `forward` signature (line 31): `Tensor(.{ .batch, .in })` in,
`Tensor(.{ .batch, .class })` out.

**Overlay:** "`Tensor(.{ .batch, .in })` ≠ `Tensor(.{ .in, .batch })`" ·
small credit line: "tagged-tensor approach inspired by ZML (README.md:386)".

### [1:00–1:32] Contraction by name

**VO:** Each dot names the axis it consumes: contract in, contract h1. Not
"the last axis of A against the first of B" — you say which axis, by name,
and a comptime function computes the result's names while the compiler runs.
And look where that function sits: in the return type of dot itself. The type
of the answer is computed from the names in the question.

**Visual:** Code shot: `README.md:31–40`, highlighting the two `dot` calls
and their comments (`contract .in -> .{ .batch, .h1 }`); then cut to
`src/ag/tensor.zig:3747`, the real `dot` signature, with the return-type
expression `!Tensor(dotResultTags(...))` highlighted.

**Overlay:** "Result tags computed at compile time — in return type position."

### [1:32–2:14] The error you came for

**VO:** So let's break it. A contraction over k — but the right operand has
no k axis anywhere. Here is the compiler's verdict, verbatim: error, tensor
tag not found. Read the notes bottom-up: our call site instantiated dot;
dot's return type called dotResultTags; its return type demanded k exist in
both operands. It doesn't. Compilation over. No wrong-shaped tensor was ever
constructed — the program that would compute the wrong shape is not a
program. This is a shape bug that never ran.

**Visual:** Code shot: the chapter's `bad()` snippet (§4.7 course code:
`a: Tensor(.{ .m, .k })`, `b: Tensor(.{ .n, .j })`, `a.dot(ctx, b, .k)`) with
the `// no .k axis anywhere` comment highlighted; then a terminal-style
render of the chapter's verbatim compiler trace (§4.7), animated bottom-up:
call site → `dot` return type (src/ag/tensor.zig:3747) → `dotResultTags` →
`src/tags.zig:187: error: tensor tag not found`.

**Overlay:** "Zig 0.16.0 — trace verbatim from the chapter" · "a shape bug
that never ran".

### [2:14–2:44] Names at comptime, sizes at runtime

**VO:** Now the honest boundary, because overselling this is easy. Names and
ranks are compile-time; sizes are runtime. These two tensors agree on names
and disagree on sizes — a d of three meets a d of two. Both calls compile,
and both fail at runtime with error ShapeMismatch. That's the loud kind of
bug. The types spend themselves on the quiet kind — where sizes match and
meaning doesn't.

**Visual:** Code shot: the machine-verified snippet quoted in chapter §4.4
(from docs/REFERENCE.md §7.7): `test "incompatible dims fail with
ShapeMismatch"` — highlight `.{3}` vs `.{2}`, then the two
`expectError(error.ShapeMismatch, ...)` lines.

**Overlay:** "Names: compile time. Sizes: runtime (`error.ShapeMismatch`)."

### [2:44–3:00] The Fortran rhyme

**VO:** It's an old discipline made new. In Fortran, real A of n and m told
you the rank in the program text. Here the names live in the text too — and
compile away entirely. Next time: the operation library those names direct.

**Visual:** Quote card: README.md:56–60 ("The shape discipline lives in the
program text, the way it did in Fortran — `real A(n,m)` told you the rank
before you read a single loop..."); brief cut to the chapter's §4.3 mini test
`"instances carry no tags at runtime"` (`@sizeOf` equality) as the
compile-away proof; end card: "Full chapter:
`docs/course/04-axes-with-names.md`", teasing Video 05.

**Overlay:** "`real A(n,m)`" → "`Tensor(.{ .batch, .in })`" · "full
chapter in `docs/course/`" · "Next: The operation library".

## Asset list

- **Code shots (repo):**
  - `README.md:24–41` — `Model` struct + `forward` (segments 2 and 3).
  - `README.md:56–60` — Fortran shape-discipline paragraph (segment 6 quote
    card).
  - `src/ag/tensor.zig:3747` — the `dot` signature with `dotResultTags` in
    return type position (segments 3 and 4).
  - `src/tags.zig:185–188` — `tagIndexOrCompileError` with the
    `"tensor tag not found"` message (optional inset during segment 4).
  - `src/tags.zig:386–392` — `dotResultLen` (optional inset during segment 3
    if pacing allows).
- **Code shots (chapter course code / quoted snippets, from
  `docs/course/04-axes-with-names.md`):**
  - §4.7 `bad()` snippet and the verbatim compiler trace below it (segment
    4). Render the trace as a styled terminal frame from the chapter text —
    do not retype or abbreviate it. If re-recording live instead: the chapter
    notes it was produced with `zig test` driven by hand against the tree
    with Zig 0.16.0, via a scratch `build_options` stub (scalar backend, no
    BLAS); reproduce that setup or use the rendered frame.
  - §4.4 snippet `test "incompatible dims fail with ShapeMismatch"` (segment
    5; source docs/REFERENCE.md §7.7, machine-verified by the
    `zig build snippet-check` CI gate).
  - §4.3 mini test `"instances carry no tags at runtime"` (segment 6 inset).
- **Diagram to draw:** segment 1 — wrong-way matmul yielding a
  plausible-shape grid of wrong numbers beside a flat loss curve (one still,
  light animation).
- **External downloads:** none. No model weights needed for this episode.

## Production notes

- **Tone:** warm, concrete, no hype. The showcase is the compiler trace —
  let it breathe on screen; the bottom-up note-chain walk is the money shot.
  Do not speed past segment 5: the honesty beat (sizes still fail at
  runtime) is mandatory, not filler.
- **Caveats that MUST stay attached:** the trace overlay "Zig 0.16.0 —
  trace verbatim from the chapter"; the ShapeMismatch snippet is
  machine-verified (REFERENCE.md
  §7.7). If any wording drifts, keep this framing: Fucina's public API is
  explicitly young — every signature shown is today's code, not a frozen
  contract (chapter §4.1). No performance numbers appear in this episode;
  do not add any.
- **Must not change:** the compiler trace text (verbatim or not at all); the
  claim pairing "names/ranks/dtype = comptime, sizes = runtime" (chapter
  §4.4, quoting REFERENCE.md §3.1); the ZML credit (README.md:386–387) —
  it must appear on screen somewhere: keep it in the segment-2 overlay, or
  move it to the segment-6 end card if that overlay is reworked.
- **Trim order if long:** first the segment-6 `@sizeOf` inset (keep the
  Fortran quote card), then the segment-3 `dotResultLen` inset, then shorten
  the segment-1 diagram animation to a still. Never trim segments 4 or 5.
- **Spoken forms:** read `.in` as "in", `dot(ctx, &m.w1, .in)` as "dot,
  contracting in"; read `error.ShapeMismatch` as "error ShapeMismatch";
  read `real A(n,m)` as "real A of n and m".
