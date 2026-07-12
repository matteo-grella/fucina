# Video 01 — Just enough Zig (3:00)

*Series: Forging Deep Learning in Zig · Source: ../01-just-enough-zig.md*

## Logline

The handful of Zig features that make a tensor library readable — allocators
as parameters, `defer` lifetimes, errors as values, and functions that return
types — shown through the library's real 27-line smoke test, ending with the
compiler itself enforcing Fucina's API policy.

## Takeaways

1. In Zig, memory and failure are visible in the code: allocators are
   parameters, every acquire has a `defer` release, and errors are values in
   closed sets — the leak checker is just `zig build test`.
2. Types are comptime values, so `fucina.Tensor(.{ .batch, .in })` is a
   *function call that returns a type* — contracting an axis a tensor doesn't
   have fails at compile time (names are compile-time; sizes stay runtime).
3. The compiler can enforce architecture: one line (`pub const Tag =
   @TypeOf(.tag)`) powers the whole named-axis system, and a `@compileError`
   guard makes exporting the internal raw tensor a build failure with a
   written explanation.

## Script

### [0:00–0:27] Twenty-seven lines

**VO:** This program builds two tensors, multiplies them, computes a loss,
and asks autograd for a gradient — checked to one part in a million.
Twenty-seven lines of Zig. No framework, no graph builder, no device flags.
It's Fucina's canonical smoke test, and CI runs it against the real library
on every push. To read it, you need a handful of Zig features. Here they are.

**Visual:** Full-screen code shot of the first program —
`docs/REFERENCE.md` lines 248–276 (the fenced `test "first program"` block,
§1.4). Slow scroll top to bottom; briefly highlight the final two
`expectApproxEqAbs` lines (tolerance `1e-6`) on "one part in a million".

**Overlay:** `27 lines — runs against the real library in CI (zig build snippet-check)`

### [0:27–1:00] Memory is a parameter

**VO:** First: memory is a parameter. Zig has no garbage collector and no
hidden malloc. Any function that allocates takes an allocator argument, and
the caller decides which one. Tests run under the standard testing allocator,
which fails the test if a single byte is still allocated at the end. So leak
detection isn't a tool you run — it's what the test suite already does. And
every tensor pairs its creation with a defer: release written next to
acquire.

**Visual:** Zoom into the same snippet: highlight `const alloc =
std.testing.allocator;` and `ctx.init(alloc);`, then each `defer x.deinit();`
line lighting up in sequence. Cut to `examples/spirals.zig` lines 327–330 —
the demo's `DebugAllocator` with `defer if (gpa.deinit() == .leak)
@panic("leak");` — on "fails the test".

**Overlay:** `a leaked byte = a failed test`

### [1:00–1:24] Errors are values

**VO:** Second: errors are values. There are no exceptions. A raw tensor
operation can fail in exactly seven ways — one closed error set, and that's
the complete list. Every fallible call is marked with try, so the failure
path of the whole program is legible at a glance. A mismatched matmul has no
unchecked escape hatch to fly through.

**Visual:** Code shot of `src/tensor.zig` lines 11–19 (`pub const TensorError
= error{ ... }`), the seven names counted off one by one. Quick cut back to
the first program with all twelve `try` keywords highlighted at once.

**Overlay:** `7 names. That's the whole failure surface.`

### [1:24–1:59] Functions that return types

**VO:** Now the big one. In Zig, types are compile-time values, so a function
can return one. That is the entirety of Zig's generics — and it's the
library's public API. Tensor is a function: call it with axis names, and it
builds a struct type during compilation. Batch-by-in and in-by-out are
different types. Contract an axis a tensor doesn't have, and the program
doesn't crash — it doesn't compile. One honest caveat: the types carry names,
not sizes. Size mismatches still surface at runtime.

**Visual:** Code shot of `src/ag/tensor.zig` lines 185–190 (`pub fn
Tensor(comptime tags_spec: anytype) type { ... }`). Then a split-screen
diagram: `Tensor(.{ .batch, .in })` and `Tensor(.{ .in, .out })` as two
distinct boxes with a ≠ between them; a third line `x.dot(&ctx, &w, .out)`
struck through with a compile-error glyph (one diagram, drawn from the
chapter's §1.11).

**Overlay:** `Tensor(.{ .batch, .in }) ≠ Tensor(.{ .in, .out })` ·
`names: compile-time — sizes: runtime`

### [1:59–2:31] The compiler as enforcer

**VO:** Two lines show how far this goes. This one-liner captures the type of
enum literals — the type of names themselves. The whole named-axis system
stands on that line, and the docs are plain: it compiles down to zero runtime
tagging cost. And here, the public root inspects its own declarations: if
anyone ever exports the internal raw tensor type, every build fails, with a
written explanation. Not a comment asking politely — API policy, enforced by
the compiler.

**Visual:** First: `src/tags.zig` line 4 (`pub const Tag = @TypeOf(.tag);`)
alone, large, centered. Then cut to `src/fucina.zig` lines 28–42: the
ownership comment plus the `comptime { if (@hasDecl(@This(), "RawTensor"))
@compileError(...) }` guard; highlight the `@compileError` string as it's
described.

**Overlay:** `pub const Tag = @TypeOf(.tag);` → then
`API policy = a compile error with an explanation`

### [2:31–3:00] Light the forge

**VO:** Try it. One compiler — exactly Zig zero-point-sixteen-point-zero —
and nothing else to install. Clone, then: zig build test. Nine test roots —
the tensor core, the LLM stack, seven examples — no model files needed, and a
leaked byte fails the build. Green means your forge is lit. The chapter
unfolds all of this, feature by feature. Next time: just enough machine
learning — what those twenty-seven lines were actually computing.

**Visual:** Terminal recording: `zig version` (prints `0.16.0`), then
`git clone https://github.com/matteo-grella/fucina`, `cd fucina`,
`zig build test` — time-lapse the build, land on the green/quiet success
state. End card with the series title and next-episode tease.

**Overlay:** `zig build test — 9 test roots, no model files` → end card:
`Next: 02 — Just enough machine learning`

## Asset list

**Code shots (repo files, exact ranges):**
- `docs/REFERENCE.md` lines 248–276 — the `test "first program"` snippet
  (§1.4; identical to the chapter's §1.2 block at
  `docs/course/01-just-enough-zig.md:78-106` — REFERENCE.md is used because
  it is the CI-verified original). The "27 lines" are the code between the
  fences (lines 249–275, blank lines included), and that code contains
  exactly twelve `try` keywords — verified, so the segment-3 highlight count
  matches the on-screen snippet. Segments 1–3.
- `examples/spirals.zig` lines 327–330 — `DebugAllocator` + `@panic("leak")`.
  Segment 2.
- `src/tensor.zig` lines 11–19 — `TensorError` (seven members). Segment 3.
- `src/ag/tensor.zig` lines 185–190 — `pub fn Tensor(...) type`. Segment 4.
- `src/tags.zig` line 4 — `pub const Tag = @TypeOf(.tag);`. Segment 5.
- `src/fucina.zig` lines 28–42 — RawTensor comment + `@compileError` guard.
  Segment 5.

**Terminal recordings (exact commands, run in repo root):**
- `zig version` — must print `0.16.0`.
- `git clone https://github.com/matteo-grella/fucina` + `cd fucina` +
  `zig build test` — passes on a fresh clone with no model assets
  (asset-dependent suites skip via `error.SkipZigTest`). Time-lapse the run.

**Diagrams (1):**
- Two-boxes-plus-≠ diagram: `Tensor(.{ .batch, .in })` vs
  `Tensor(.{ .in, .out })` as distinct types, with `x.dot(&ctx, &w, .out)`
  shown as a compile error (content from chapter §1.11 and the §1.2 ML note).

**External downloads:** none. No model files are needed anywhere in this
episode — that is part of the point of `zig build test`.

## Production notes

- **Tone:** warm and concrete; the episode's engine is "read real code,
  feature by feature". No hype — the strongest claims are quoted ones.
- **Numbers and their sources (must survive the edit):** "27 lines",
  "nine test roots", "seven error names", "twelve `try`s", "0.16.0", and
  "zero runtime tagging cost" all come from the chapter /
  `docs/REFERENCE.md`; "one part in a million" is the `1e-6` tolerance
  visible in the on-screen code. No benchmark figures appear in this episode,
  so no machine/date caveats are needed — do not add performance numbers in
  refinement.
- **Caveat that must not be cut:** segment 4's "names are compile-time,
  sizes are runtime" line. Without it the compile-time-shapes claim
  overpromises; the chapter states this caveat explicitly (§1.11).
- **Must not change:** the two showcase code shots (`src/tags.zig:4` and the
  `src/fucina.zig` guard) and the fact that the first program is shown as
  *the repo's own CI-verified snippet*, not a mock-up.
- **If the cut runs long, trim in this order:** (1) the
  `examples/spirals.zig` cutaway in segment 2 (keep the testing-allocator
  line); (2) the twelve-`try` highlight cut in segment 3; (3) shorten the
  clone/build time-lapse. Do not trim the caveat line or the teaser.
- **Pacing:** VO is 438 words ≈ 146 wpm over 3:00. Segment 5 needs the two
  code shots to land exactly on "this one-liner" and "and here" — sync the
  cuts to those words.
- Series-level candor (one person's effort with agentic-coding assistance;
  AI-generated course) is carried by episode 00 per the chapter-00 framing;
  this episode makes no claims that require restating it, but the end card
  may carry the standard series footer if one exists.
