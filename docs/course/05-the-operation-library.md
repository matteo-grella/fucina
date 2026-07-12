# Chapter 05 — The operation library

*Part II — The tensor core*

You now have tensors ([Chapter 3](03-tensors-from-scratch.md)) and axes with
names ([Chapter 4](04-axes-with-names.md)). What you do not yet have is
*verbs*. A deep learning library is, more than anything else, a vocabulary of
operations: add and multiply, activate, normalize, contract, attend, gather,
lose. This chapter walks that vocabulary as Fucina implements it — and because
the ops *are* the concepts, it doubles as a guided tour of deep learning's own
vocabulary. Along the way you will meet the runtime every op runs through
(`ExecContext`), the one contract they all share, and a design stance —
determinism as a promise, not an accident — that quietly shapes everything.

One orientation note first: everything in this chapter is *eager*. When you
call an op, the kernel runs, and you get a finished tensor back. There is no
graph object, no session, no compile step, no lazy evaluation — "an op call
runs the kernel and returns an owned result immediately"
(docs/REFERENCE.md §6). You can single-step an entire transformer forward
pass in a debugger, one op at a time. Hold on to that; it is the most
load-bearing design fact in the library.

> **ML note** — PyTorch is eager too; what Fucina drops is everything
> *around* the eager core — no global default device, no implicit engine in
> the background, no graph-capture modes. Coming from compiler-style
> frameworks (JAX under `jit`), the difference is bigger: there is no
> deferred program to inspect or optimize. What you read is what runs.

## 5.1 ExecContext: the world in a struct

Every op in Fucina takes `ctx: *ExecContext` as its first runtime argument.
The context is the entire runtime, in one value you create and destroy
yourself: it owns the allocator wrapper, the backend instance, the
transient-buffer pool, the lazily created worker team, and the exec-scope
stack (docs/REFERENCE.md §6; the struct lives in `src/exec.zig`). There are
no globals anywhere: two contexts are two fully independent worlds, and
destroying one cannot disturb the other.

Two of the context's possessions deserve a first look now (both get a full
chapter's attention elsewhere).

**The buffer pool.** You met the memory model in
[Chapter 3](03-tensors-from-scratch.md): every transient tensor draws its
storage from a per-context `BufferPool`, and `deinit` is *the recycling
driver, not a naive free* — releasing a tensor returns its buffer to a free
list, and a same-sized successor gets the very same address back, warm in
cache. That is asserted behavior, not folklore — Chapter 3 pins it with the
pointer-equality recycling test (machine-verified from
docs/REFERENCE.md §6.2). What matters for this chapter: op results come
from the pool; kernels never allocate.

> **Zig note** — Note the creation idiom every runnable snippet in this
> chapter opens with: `var ctx: ExecContext = undefined; ctx.init(alloc);`
> — not `var ctx =
> ExecContext.init(alloc);`. The context is **self-referential**
> (`rt.allocator` is a fat pointer into `rt.thread_safe_allocator`, a field
> of the same struct), so if `init` returned a value, the compiler would
> copy the struct and the interior pointer would dangle. Taking
> `self: *ExecContext` and returning `void` forces initialization *in
> place*, at the address where the context will live — and it must never be
> moved or copied afterwards (docs/REFERENCE.md §6.1). This "pinned struct"
> idiom appears throughout systems Zig. Its mirror image lives in `deinit`
> (`src/exec.zig`): after teardown, `self.* = undefined;` — so a
> use-after-deinit trips loudly in safety builds instead of reading stale
> state.

The pool also gives you leak detection for free: `BufferPool.deinit` — run by
`ExecContext.deinit` — asserts that no pooled buffer is still outstanding, so
a tensor leaked past context teardown fails an assertion in safety builds
instead of silently leaking (docs/REFERENCE.md §6.1).

**The worker team.** CPU kernels parallelize over a persistent fork-join
team owned by the context — created lazily, on the first op big enough to
want it: "A context that only ever runs small/serial ops never starts a
thread" (docs/REFERENCE.md §6.6). The team's mechanics are
[Chapter 6](06-going-fast-on-cpus.md)'s material.

Finally, layering. `ExecContext` is a thin forwarding facade over `Runtime`
(`src/exec/runtime.zig`); the actual op implementations live in leaf modules
under `src/exec/` — `elementwise.zig`, `matmul.zig`, `softmax.zig`,
`moe.zig`, and friends — which receive `*Runtime` explicitly and never
import `exec.zig` back. Only `src/exec.zig` is public API: "the domain
modules under `src/exec/` it forwards to are not public API"
(docs/REFERENCE.md §6.4). They are excellent *reading* — we will quote them
— but you write programs against `ExecContext` and the tagged facade, not
against the leaves.

## 5.2 The common op contract

Fucina's op surface is large — the tagged facade in `src/ag/tensor.zig` spans
well over a hundred methods, and the raw surface beneath it roughly 300
`pub fn`s in `src/exec.zig` (docs/REFERENCE.md §6.4). What makes it learnable
is that every single op honors one contract (docs/REFERENCE.md §4.1):

1. **Signature shape.** Ops are methods on `Tensor(tags)` taking
   `ctx: *ExecContext` first. Axes are chosen by **comptime tag**: misnaming
   a tag is a *compile error*, never a runtime error; shape problems the
   type system cannot see are recoverable `TensorError`s (`ShapeMismatch`,
   `InvalidShape`, `IndexOutOfBounds`, …) — the two-level shape discipline
   from [Chapter 4](04-axes-with-names.md): names at compile time, extents
   at runtime.
2. **Ownership.** Each op allocates and returns a **new owned tensor**; the
   caller `deinit`s it. Operands are borrowed via `*const` and never
   consumed (two documented exceptions below).
3. **Gradients.** A backward record is attached iff at least one operand
   `requiresGrad()` and gradients are globally enabled. That machinery is
   [Chapter 7](07-autograd.md)'s story; here we only note which families are
   no-grad by design.
4. **Thread-safety.** One context, one thread. Kernels fan work out to the
   context's team internally and join before returning.

A note before any signature appears: Fucina's public API is young and
explicitly unstable — no semver, no package manifest. Every signature in this
chapter is *today's code*, cited by file, not a frozen contract.

Behind the contract sits an equally uniform implementation shape:
**validate once at the facade, then dispatch to small unchecked kernels.**
The facade checks everything checkable, exactly once; below it, kernels are
tight loops that trust their inputs and never re-validate. Three views of
the real thing show the whole architecture. First, the facade methods are
startlingly small. `add`, in its entirety (from `src/ag/tensor.zig:1155`):

```zig
pub fn add(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags)) {
    return pointwise(.add, self, ctx, other);
}
```

The return *type* is computed at comptime from both operands' tag sets —
that is Chapter 4's `pointwiseResultTags` doing its job. And the full
pattern, for an op that carries its own backward record (from
`src/ag/tensor.zig:1171`):

```zig
pub fn scale(self: *const Self, ctx: *ExecContext, scalar_value: f32) !Self {
    var value = try ctx.scale(self.asRawTensor(), scalar_value);
    errdefer value.deinit();
    return finishOp(tags, ctx, value, self.requiresGrad(), ScaleBackward(tags), .{ ctx.allocator, self.grad_state, scalar_value });
}
```

Run the kernel, `errdefer` the value, hand off to `finishOp` naming the
backward type. Every differentiable op in the library ends in one of two
shared tails — `finishOp` (attach a backward record if wanted) or
`finishNoGrad` — so the contract is implemented in exactly one place, not
re-derived per op (`src/ag/tensor.zig`).

Second, the lowering tier shows "validate once" concretely. This is the
entire tag-broadcast pointwise lowering (from `src/tagged.zig:53`):

```zig
/// Tag-driven broadcasting pointwise op: broadcasts both operands to the
/// pointwise result tags, then dispatches the rank-matched kernel.
pub fn pointwise(
    comptime op: PointwiseOp,
    comptime left_tags: anytype,
    left: *const RawTensor,
    ctx: *ExecContext,
    comptime right_tags: anytype,
    right: *const RawTensor,
) !RawTensor {
    try validateTensorRank(left_tags, left);
    try validateTensorRank(right_tags, right);
    const result_tags = pointwiseResultTags(left_tags, right_tags);
    const result_shape = try broadcastResultShape(result_tags, left_tags, left, right_tags, right);

    var left_view = try broadcastTensorTo(left_tags, left, result_tags, result_shape);
    defer left_view.deinit();
    var right_view = try broadcastTensorTo(right_tags, right, result_tags, result_shape);
    defer right_view.deinit();

    return switch (op) {
        .add => ctx.addRank(rawRank(result_tags.len), &left_view, &right_view),
        .sub => ctx.subRank(rawRank(result_tags.len), &left_view, &right_view),
        // ... abridged: .mul, .div, .max, .min follow the same shape
    };
}
```

Validate shapes, compute the result tags (comptime), build **zero-copy
broadcast views** (a broadcast is a zero-stride view, never a materialized
copy), then dispatch to rank-specialized kernels that *assume* shape
correctness. Below this line, nothing re-checks anything.

Third, the kernels themselves classify layout exactly once and pick a fast
path. The dispatch key is a four-way enum (from `src/exec.zig:47`):

```zig
pub const LayoutClass = enum {
    contiguous,
    scalar,
    tail_broadcast,
    arbitrary,
};
```

`contiguous` gets the vectorized hot loop; `scalar` and `tail_broadcast`
(a bias-style operand: broadcast over the leading axes, with a contiguous
trailing block) get their own specializations;
`arbitrary` gets the honest strided fallback. And one more discipline holds
across the whole kernel tier: **kernels never allocate** — every transient
buffer they need flows through the context's pool (docs/REFERENCE.md §6.5).
What the kernels look like inside — `@Vector`, thresholds, the worker team —
is [Chapter 6](06-going-fast-on-cpus.md).

> **Zig note** — Notice `other: anytype` in `add`. Zig has no overloading;
> `anytype` plus comptime inspection of `@TypeOf(other)` is how one entry
> point accepts a tensor value or a pointer — and, for ops like `dot`, how
> an f32, f16, or quantized right-hand side each routes to a different
> implementation *at compile time*. Option arguments use the same trick:
> `softmax(ctx, .src, .{ .scale = s })` takes a comptime-validated struct
> literal, and a misspelled field is a compile error, not a silently
> ignored setting (docs/REFERENCE.md §4.10).

## 5.3 Ownership in practice: deinit-ASAP, `replace`, and exec scopes

The contract says every result is caller-owned. In inference code the idiom
is *deinit-ASAP*: release each intermediate the moment you are done with it,

```zig
var y = try x.relu(&ctx);
defer y.deinit();
```

and the pool turns that discipline into an O(1) working set — a chain of
same-shaped ops recycles the same couple of buffers over and over, warm in
cache (docs/REFERENCE.md §6.2).

Loops that *carry* a value — a residual stream advancing through transformer
blocks — get a dedicated helper, `ctx.replace`, which frees the old value and
rebinds in one statement, error-safely:

```zig
test "ctx.replace advances a carried tensor" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer x.deinit(); // runs on whatever x holds at scope exit
    for (0..3) |_| {
        // frees the old x and rebinds — on error the old x stays valid
        x = try ctx.replace(x, x.scale(&ctx, 2.0));
    }
    try std.testing.expectEqualSlices(f32, &.{ 8, 16 }, try x.dataConst());
}
```

*(machine-verified snippet from docs/REFERENCE.md §6.2)*

Training breaks deinit-ASAP: every intermediate between the parameters and
the loss must stay alive until `backward()` runs. **Exec scopes** solve this
without changing your forward code. While a scope is open on the context
(`const scope = ctx.openExecScope(); defer ctx.closeExecScope(scope);`),
every facade-op result is adopted by the scope; the value you receive is a
borrow whose `deinit` is a safe no-op, and `closeExecScope` releases
everything at once, newest first. The canonical training step
(docs/REFERENCE.md §6.3) divides ownership three ways: *parameters* are
caller-owned (`defer w.deinit()` outside the loop), *intermediates* are
scope-owned borrows (no defers at all), and *fetched gradients* are
caller-owned again. Because `deinit` on a scope-owned borrow is a no-op, the
*same* defer-deinit forward code runs correctly both ways — write the
forward once, train it by wrapping a scope around it. Why training needs
this at all (single-owner graph nodes) is [Chapter 7](07-autograd.md)'s
subject, and [Chapter 8](08-training.md) runs the full loop.

Be clear about what scopes are *not*: they are a training tool, **not an
inference optimization**. A held scope inverts the pool discipline — instead
of recycling a couple of warm buffers it keeps every intermediate live until
close: "measured 2 vs 32 distinct buffers on a 32-op 1 MiB chain"
(docs/MEMORY-MODEL.md §5). For pure inference, deinit-ASAP with no scope is
the discipline.

Four ownership gotchas, all documented, all worth memorizing now
(docs/REFERENCE.md §4.1, §6.3):

- **Scope borrows die at close** — use one after `closeExecScope` and it is
  use-after-free.
- **Typed constants stay caller-owned even under a scope**: `.bool` masks,
  i64 index outputs, explicit constructors, fetched gradients. The classic
  trap is `topK` under a scope — `values` is a scope borrow, `indices` is
  caller-owned.
- **Composed ops require a scope under gradients** (`nllLoss`,
  `l2Normalize`, `cosineSimilarity`, `stack`, `einsumMany`, …): they build
  function-local graph nodes only a scope can own —
  `error.ActiveExecScopeRequired` otherwise; no-grad use works unscoped.
- **The two consuming ops refuse borrows**: consuming a scope borrow would
  double-free at close — `error.ActiveExecScopeUnsupported`.

One layering remark to file away for Chapter 7: the exec layer "deliberately
knows nothing about autograd types; the ag facade stores its backward nodes
in that payload. The user scopes the execution; that, in turn, is what
enables autograd on top" (comment in `src/exec.zig:141-145`). A scope holds a
type-erased `*anyopaque` plus a destructor pointer per adopted value — exec
owns things it cannot name.

## 5.4 Pointwise ops and broadcasting: the ground floor

The simplest family — `add`, `sub`, `mul`, `div` — carries the library's
most useful semantic decision: **broadcasting by tag name, not by position.**
The result tag set is `self`'s tags in order, then `other`'s tags that
`self` lacks; per shared tag the sizes must be equal or one of them 1; a tag
missing from one operand behaves as size 1. No `unsqueeze`, no `expand`, no
counting dims from the right:

```zig
test "pointwise add broadcasts by tag" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var b = try fucina.Tensor(.{.col}).variableFromSlice(&ctx, .{2}, &.{ 10, 20 });
    defer b.deinit();

    var y = try x.add(&ctx, &b); // result tags .{ .row, .col }; b broadcasts over .row
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 13, 24 }, try y.dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gb = (try b.grad(&ctx)).?; // broadcast VJP: gradient reduced back to .{ .col }
    defer gb.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 2 }, try gb.dataConst());
}
```

*(machine-verified snippet from docs/REFERENCE.md §4.2)*

> **ML note** — This tiny test is half of a neural network already. A
> "linear layer" is a matrix multiply plus a **bias**: per-feature offsets
> added to every row of a batch, and broadcasting lets a `[features]` bias
> meet a `[batch, features]` activation without copying it — a zero-stride
> *view*. The last four lines show the flip side, which trips up everyone
> once: a gradient flowing into a broadcast operand must be **sum-reduced**
> over the broadcast axes (the bias served 2 rows, so its gradient is the
> sum of 2 row-gradients — here `{2, 2}`). The tag rule makes both
> directions automatic.

Three more binary ops share the tag-broadcast rule (docs/REFERENCE.md §4.2):
`maximum`/`minimum` follow torch's NaN-propagating semantics (a NaN in
either operand wins — deliberately *not* the IEEE rule Zig's bare `@max`
follows), and `pow` follows `std.math.pow` domain semantics with torch's
gradient conventions.

**Scalar variants** cover the ubiquitous tensor-with-a-number cases without
building a scalar tensor first: `scale(ctx, s)`, `addScalar`, `subScalar`,
`divScalar`, `powScalar` — all differentiable, all returning new tensors
(docs/REFERENCE.md §4.3).

**In-place and no-grad helpers** exist for inference hot paths, where the
"always a new tensor" rule would cost real bandwidth (docs/REFERENCE.md
§4.3): `addAxisVectorInPlace` (bias-add along the last axis, mutating
`self`), `addAxisVectorUnaryInPlace` (fused bias + activation),
`addScaledInPlace` (`self += alpha·other`), and the two *consuming* ops
`takeAddNoGrad`/`takeScaleNoGrad`, which take ownership of `self` and reuse
its storage when they can. All of them reject grad-requiring operands with
`error.UnsupportedGradient` — mutation and recorded history do not mix. For
a *trainable* bias, use broadcast `add` as in the test above.

## 5.5 The activation zoo

Between every pair of linear layers, a network needs a nonlinearity —
otherwise a stack of matrix multiplies collapses into one matrix multiply.
Fucina's elementwise nonlinearities live in one closed kernel enum,
`exec.UnaryOp` — twenty-seven members today, most with a direct method alias
(`x.relu(&ctx)` is `x.unary(&ctx, .relu)`); the full table is
docs/REFERENCE.md §4.4. A curated sample, because the zoo *is* a history of
deep learning:

| Op | What it is | Why it exists |
|---|---|---|
| `.relu` | `max(x, 0)` | the classic; cheap, sparse, trains deep nets |
| `.sigmoid`, `.tanh` | squashers | the pre-relu era; still gates and heads |
| `.silu` | `x·sigmoid(x)` | the modern default in LLM FFNs |
| `.gelu` | tanh-approximated GELU | the transformer-era default |
| `.gelu_erf` | exact-erf GELU | when parity with an exact reference matters |
| `.gelu_quant` | f16-rounded tanh-GELU | bit parity with ggml's `GGML_GELU_FP16` table |
| `.softcap_30` | `30·tanh(x/30)` | Gemma's logit softcap |
| `.fast_tanh` | rational tanh approximation | the NAM guitar amp's realtime path ([Chapter 10](10-the-guitar-amp.md)) |

Notice what several rows have in common: they exist for *interoperability*.
When you port a model, "gelu" is not one function — it is a family of
approximations, and matching the reference's exact variant is the difference
between bitwise parity and mysterious drift. Fucina names the variants
instead of pretending they are the same op.

Alongside the enum: `leakyRelu(ctx, negative_slope)` and
`clamp(ctx, min, max)` (parameterized, differentiable), and
`dropout(ctx, p, seed)` — deferred to §5.12, because its design is the
determinism story in miniature.

**Gated activations** are the modern FFN's shape (docs/REFERENCE.md §4.5).
`exec.GatedOp` is `{ .glu, .swiglu, .geglu, .swiglu_clamp10 }`; the
two-operand form computes `self · act(other)` — the **second** operand is
the gate — so `up.swiglu(&ctx, &gate)` is `up · silu(gate)`. `splitGated`
fuses the common storage trick where the up- and gate-projections are one
concatenated tensor halved along an axis, and even the gate-half
conventions are pinned deliberately (`.swiglu` gates with the *first* half,
`.glu` with the *second*, matching ggml) — another parity decision made
once, in the library, instead of per port. `.swiglu_clamp10` (DeepSeek V4's
clamped SwiGLU) is inference-only and exists for the MoE entries; `gated`
and `splitGated` reject it at compile time.

> **ML note** — Why gates? A plain FFN computes `W2·act(W1·x)`. A *gated*
> FFN computes `W2·(act(Wg·x) ⊙ (Wu·x))` — one projection decides "how
> much", the other "what". SwiGLU (SiLU gate) is the empirical winner used
> by Llama-family models; GeGLU (GELU gate) is Gemma's choice. You will
> build one for real in [Chapter 12](12-a-transformer-from-scratch.md).

Why is `UnaryOp` a *closed* enum? A finite kernel table stays auditable and
optimizable — every member has vetted scalar and SIMD legs. Extensibility
lives one level up, in the elemental tier (§5.13), which lifts *your* scalar
function over the same machinery.

## 5.6 Masks, comparisons, and conditionals

Deep learning code is full of decisions applied elementwise: which attention
positions are visible, which tokens are padding, which logits to suppress.
The vocabulary for that is masks (docs/REFERENCE.md §4.6):

- `compare(ctx, op, other)` produces a `.bool` tensor (`op` is one of
  `.eq .ne .lt .le .gt .ge`; `other` a same-tagged tensor or scalar, chosen
  at comptime). Constant by design — no gradient flows through a
  comparison. NaN follows IEEE; integer tensors compare natively, exact at
  any magnitude.
- `where(ctx, cond, other)` — `cond[i] ? self[i] : other[i]`,
  differentiable in `self` and `other`; the condition is a non-grad mask.
- `maskedFill(ctx, mask, value)` — `mask[i] ? value : self[i]`,
  differentiable in `self` with the gradient zeroed where filled.
- `logicalAnd/Or/Xor/Not`, `isnan/isinf/isfinite`, `any/all` complete the
  set.

The reference's own demonstration builds relu out of masks:
`x.maskedFill(&ctx, &neg, 0)` where `neg = x.compare(&ctx, .lt, 0)` — and
counting the mask with `neg.sumAll(&ctx)` returns an **i64** scalar, the
mask-counting idiom (docs/REFERENCE.md §4.6; Exercise 1 has you write it).
Two details from that snippet matter beyond the toy. First, the ownership
fine print from §5.3: `.bool` results are typed constants, so they stay
**caller-owned even under an exec scope** — the mask's `deinit` is
load-bearing in training code where the f32 results around it would be scope
borrows. Second, the dtype is checkable at compile time:
`comptime std.debug.assert(@TypeOf(neg).dtype == .bool)` — the mask-ness of
a tensor is part of its *type*, exactly like its tags.

## 5.7 Reductions and scans

Reductions collapse an axis; in Fucina they also collapse the *type*:

```zig
var s = try x.sum(&ctx, .col); // x: Tensor(.{ .row, .col }) -> s: Tensor(.{ .row })
```

The compiler now knows the column axis is gone — pass `s` where a
`.{ .row, .col }` tensor is expected and the mistake is a compile error, not
a shape crash three layers later. `sumAll` reduces everything to the scalar
`Tensor(.{})`, read with `item()`. The family (docs/REFERENCE.md §4.7):
`sum`, `mean`, `sumMany` (several tags at once), `sumAll`,
`variance(ctx, tag, ddof)`, `max`/`min` (values only — indices are §5.11's
business), `prod`, the scans `cumsum`/`cumprod` (shape-preserving prefix
sum/product), and `norm`/`normAll` with `NormOrder = { .l1, .l2, .inf }` —
all differentiable, with the core of the family machine-verified in the
reference's snippets.

> **ML note** — That odd `ddof: u1` parameter on `variance` encodes a real
> statistical fork: `ddof = 0` divides by `n` (the *biased* estimator —
> what LayerNorm uses), `ddof = 1` divides by `n − 1` (Bessel's correction —
> `torch.var`'s default). Frameworks disagree on defaults, and normalization
> layers written against the wrong one produce subtly wrong models. Making
> it a required one-bit argument forces the choice into the open.

Two behaviors here foreshadow §5.12. `max`/`min` route the gradient to the
**first** occurrence of the extremum (a pinned tie-break, matching
`torch.max` over a dim). And `cumsum`/`cumprod` are serial per row *by
default* precisely so their results are bitwise deterministic for any
thread count; the opt-in `-Dvector-scan` build flag vectorizes them with
the reassociation trade-off documented on the flag (docs/REFERENCE.md §2.2,
§4.7). Floating-point addition is not associative; this library treats the
order of summation as part of an op's contract.

## 5.8 Contraction: `dot`, `einsum`, and friends

Matrix multiplication is *the* operation of deep learning — every linear
layer, every attention score, every logit projection is a contraction. In
Fucina you name the axis you contract instead of arranging operands into
blessed layouts:

```zig
pub fn dot(self: *const Self, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag)
    !Tensor(dotResultTags(tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag))
```

*(signature from `src/ag/tensor.zig:3318`)*

At comptime every tag falls into one of three roles (docs/REFERENCE.md §4.8):
the **contract** tag (named by you, removed from the result), **batch** tags
(every other tag shared by both operands — sizes must match exactly), and
**free** tags (private to one operand). The result order is
`batch ++ left-free ++ right-free`. Because axes are named, *no transpose
calls are ever needed around `dot`* — orientation is the lowering's problem,
not yours:

```zig
test "dot with a shared batch tag lowers to bmm" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .b, .m, .k }).fromSlice(&ctx, .{ 2, 1, 2 }, &.{ 1, 2, 5, 6 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .b, .k, .n }).fromSlice(&ctx, .{ 2, 2, 1 }, &.{ 3, 4, 7, 8 });
    defer b.deinit();
    var y = try a.dot(&ctx, &b, .k); // .b is shared (batch), result .{ .b, .m, .n }
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 83 }, try y.dataConst());
}
```

*(machine-verified snippet from docs/REFERENCE.md §4.8)*

The shared `.b` tag became a batch axis automatically; in positional
frameworks that is a separate `bmm` entry point you must remember to use.
How the lowering picks kernels — zero-copy orientation selection, batch
collapse — belongs to [Chapter 4](04-axes-with-names.md); the GEMM kernels
themselves to [Chapter 6](06-going-fast-on-cpus.md).

`other`'s dtype is comptime-dispatched, and this is where frozen weights
enter the picture (docs/REFERENCE.md §4.8): an f32 RHS gets the full
two-operand backward; an f16/bf16 *constant* RHS is a frozen weight —
gradient flows to `self` only, through mixed kernels that widen in-register
(a grad-requiring 16-bit *variable* RHS still receives its own f32
gradient); a block-quantized RHS
([Chapter 11](11-model-files-and-quantization.md)) runs the quantized GEMM,
gradient to `self` only. One method, four storage worlds, zero runtime
dispatch.

**`einsum` generalizes `dot`** from one contraction tag to a whole Einstein
equation — and because operands already carry named axes, *the output tag
tuple is the equation*:

```
result[out_tags] = Σ over every tag not in out_tags of self ⊙ other
```

The rule is pure membership (docs/REFERENCE.md §4.8): shared tags kept in
`out_tags` are batch axes; shared tags dropped are contractions (any number
of them); private tags kept are free; private tags dropped are summed away.
So `a.einsum(&ctx, &b, .{.n})` with `a[.s, .k]` and `b[.k, .n]` contracts
`.k` *and* pre-sums `.s` in one call, and the result axis order is exactly
`out_tags`. `einsumMany` folds two or more operands left-to-right (with two it is just
binary `einsum`) — the reference's example is a ready-made introduction to
LoRA
([Chapter 15](15-training-llms-on-cpu.md)):

```zig
test "einsumMany: a LoRA delta as one three-operand equation" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .s, .i }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 1 });
    defer x.deinit();
    var a = try fucina.Tensor(.{ .r, .i }).fromSlice(&ctx, .{ 1, 2 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .o, .r }).fromSlice(&ctx, .{ 2, 1 }, &.{ 1, -1 });
    defer b.deinit();

    // x[s,i] · A[r,i] · B[o,r] -> [s,o], contraction order = operand order.
    var y = try fucina.einsumMany(&ctx, .{ .s, .o }, .{ &x, &a, &b });
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, -5 }, try y.dataConst());
}
```

*(machine-verified snippet from docs/REFERENCE.md §4.8)*

Internally, `dot` **is** einsum — `taggedDot` delegates to `taggedEinsum`
with the canonical dot result order as the equation (docs/REFERENCE.md §4.8).
One contraction engine, not two. And contractions are closed under
differentiation: each operand's gradient is *another einsum* (the output
gradient contracted with the other operand), so both backward branches stay
on GEMM kernels for every tag structure. That closure retired an old
broadcast-multiply backward fallback — "`zig build bench-einsum` measured
that case at two orders of magnitude" (docs/REFERENCE.md §4.8).

Rounding out the family: `matmul(ctx, other, kind, out_tags)` is the
explicit escape hatch that bypasses the tag algebra entirely — you name the
result axes and pick `.plain`/`.trans_a`/`.trans_b` yourself
(docs/REFERENCE.md §4.9). The packed-RHS entries next to it (`dotPacked`,
`rmsNormMulDotPacked`, `splitSwiGluDotPacked`, …) are **inference-only**
fused quantized GEMMs; they belong to
[Chapter 11](11-model-files-and-quantization.md)'s quantization story.

## 5.9 The transformer's verbs: softmax, norms, RoPE, attention, conv

This section is the ML vocabulary lesson proper: five op families, each the
crystallized form of a concept you will use for the rest of the course.

### Softmax: probability from scores

`softmax(ctx, tag, options)` turns arbitrary real scores along a named axis
into a probability distribution — nonnegative, summing to 1. It is how a
network *chooses*: the next token, the attended position, the class.
Numerically it uses the max-shift trick (subtract the row maximum before
exponentiating, so `exp` never overflows); the kernel's SIMD anatomy waits
in [Chapter 6](06-going-fast-on-cpus.md).

The options struct is a compressed tour of a decade of attention
engineering (docs/REFERENCE.md §4.10) — the effective pre-softmax logit is
`x·scale + slope·mask`:

- `.scale` — fold attention's `1/sqrt(d)` into the kernel, no separate pass;
- `.mask` — an **additive** tag-broadcast mask (a `[q, k]` mask serves every
  head of a `[head, q, k]` score tensor by zero-stride broadcast);
- `.max_bias` + `.head_tag` — ALiBi's per-head distance penalties;
- `.sinks` — per-head attention sinks: an extra logit that joins the
  denominator only, absorbing probability mass;
- `.causal = .{ .query_tag = ... }` — fused causal masking. On a 2×2
  all-zero score tensor, plain `softmax(&ctx, .src, .{})` gives uniform
  rows `{0.5, 0.5}`; adding `.causal` turns row 0 into `{1, 0}` — query 0
  attends source 0 only, and the masked-out tail is *exactly* 0, not
  merely tiny (pinned by the machine-verified test in
  docs/REFERENCE.md §4.10).

Two log-domain companions, `logsumexp` and `logSoftmax`, share the same
max-shifted row machinery as fused single-node kernels — though when the
next step is a loss, prefer `crossEntropy` (§5.10), which fuses further.

### Normalization: keeping activations tame

Deep stacks drift — activations grow or shrink layer by layer until
training destabilizes. Normalization layers re-center and re-scale along the
feature axis, and the two dominant recipes sit side by side
(docs/REFERENCE.md §4.11):

- `layerNorm(ctx, tag, eps, options)` — `(x − μ)/sqrt(σ² + eps)`, biased
  variance (§5.7's `ddof = 0`), optional fused affine
  (`.{ .weight = &w, .bias = &b }` — the fused kernel requires both
  together);
- `rmsNorm(ctx, tag, eps)` — `x / sqrt(mean(x²) + eps)`: no mean
  subtraction, no bias — cheaper, and the modern LLM default.

On the row `{1, 3}`, layerNorm gives `{-1, 1}` (center, then scale to unit
variance) while rmsNorm gives `x / sqrt((1 + 9)/2)` — no centering; the
reference pins both numerically (docs/REFERENCE.md §4.11).

The family's option surfaces record real interop traps: `standardizeAxis`
lets you choose where the epsilon goes — `sqrt(var) + eps` versus
`sqrt(var + eps)` — because reference models genuinely differ, and the two
placements do not produce the same numbers (docs/REFERENCE.md §4.11). Fused
variants (`rmsNormMul`, `rmsNormMulAdd`, `rmsNormMulRopeHalfPrepared`) exist
because norm → scale → next-op chains are the hottest few lines of an LLM
forward; all stay differentiable in every tensor operand, statistics
recomputed in the backward. (`rmsNormMulDotPacked`, fusing into a packed
quantized GEMM, is inference-only — §4.9's packed policy.)

### RoPE: positions as rotations

Transformers have no built-in notion of word order; positional information
must be injected. Rotary position embeddings encode position `p` by
*rotating* consecutive feature pairs by position-dependent angles — pair `i`
at position `p` rotates by `p / theta_base^(2i/d)`:

```zig
pub fn rope(self, ctx, comptime position_tag: Tag, comptime feature_tag: Tag,
            source: anytype, comptime mode: RopeMode) !Self
```

*(signature from docs/REFERENCE.md §4.12; impl `src/ag/tensor.zig`)*

`mode` picks the pairing layout (`.half` pairs feature `i` with `i + d/2`,
NEOX/Llama-style; `.interleaved` pairs adjacent features) — another interop
fork made explicit. The production path builds a `RopeTable` of factors once
with `ctx.prepareRopeTable(...)` and reuses it per layer; the table's
`feature_dim` is the authoritative rotary span (smaller than the axis =
partial rotation), and "negative positions rotate backwards, so re-roping
cached values to a new offset is a valid pattern" (docs/REFERENCE.md §4.12)
— a sentence that will make full sense when the KV cache arrives in
[Chapter 12](12-a-transformer-from-scratch.md), which owns the why and the
geometry.

### Attention, in one call

`groupedAttention` is the fused flash-style attention entry:

```zig
pub fn groupedAttention(self, ctx, k: anytype, v: anytype, kv_head_for_head: []const usize,
                        comptime out_tag: Tag, scale_value: f32, opts: anytype)
    !Tensor(.{ .seq, out_tag })
```

*(signature from docs/REFERENCE.md §4.13; impl `src/ag/tensor.zig`)*

The query *must* be tagged `.{ .seq, .head, .d }` — a compile error
otherwise — and `kv_head_for_head` maps query heads to KV heads: that is
grouped-query attention (several query heads sharing one KV head) as a plain
slice. The K/V representation is comptime-dispatched from `@TypeOf(k)`: f32
tensors (training — full q/k/v backward), f16 (decode caches — q-gradient
only), raw q8_0 blocks, and two ragged multi-stream forms (inference-only)
all route through this one name (docs/REFERENCE.md §4.13). The smallest true
statement about attention makes a satisfying test — with a single cached
position, attention over it returns exactly `v`:

```zig
var y = try q.groupedAttention(&ctx, &k, &v, &.{0}, .out, 1.0, .{});
```

*(from the machine-verified snippet in docs/REFERENCE.md §4.13)*

Everything deeper — scores, caches, masks, windows — is
[Chapter 12](12-a-transformer-from-scratch.md).

### Convolution and pooling

The 1-D family (`conv1d`, `causalConv1d`, `causalDepthwiseConv1d`,
`convTranspose1d`, …) operates on `[time, channel]` storage with PyTorch's
cross-correlation semantics; the causal variants pin the orientation ("tap
`taps−1` is the newest sample") and accept streaming state — the beating
heart of the guitar amp in [Chapter 10](10-the-guitar-amp.md). The
channel-last 2-D family (`conv2d`, `maxPool2d`, `avgPool2d`, `prelu`, …)
serves the vision stack. Both get their proper treatment later; for now,
note only that they obey the same contract as everything else — named axes,
owned results, differentiable unless documented otherwise
(docs/REFERENCE.md §4.14).

## 5.10 Losses: measuring wrongness

A loss reduces "how wrong is the model" to one differentiable number. The
centerpiece is cross-entropy — and Fucina's is *fused*: log-softmax and
negative-log-likelihood as one op, one backward record.
`crossEntropy(ctx, class_tag, labels)` takes its labels as plain host-side
`[]const usize` — no label tensor ceremony — and returns the scalar
`Tensor(.{})` (docs/REFERENCE.md §4.15). The best sanity check in the whole
library: on uniform logits over `K` classes, the loss must be exactly
`ln(K)` — the entropy of pure ignorance:

```zig
test "crossEntropy on uniform logits is ln(K)" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var logits = try fucina.Tensor(.{ .batch, .class }).zeros(&ctx, .{ 1, 4 });
    defer logits.deinit();
    var loss = try logits.crossEntropy(&ctx, .class, &.{2});
    defer loss.deinit();
    try std.testing.expectApproxEqAbs(@log(@as(f32, 4)), try loss.item(), 1e-6);

    var per_pos = try logits.crossEntropyExt(&ctx, .class, &.{2}, .{ .reduction = .none });
    defer per_pos.deinit(); // Tensor(.{ .batch }): class tag removed
    try std.testing.expectApproxEqAbs(@log(@as(f32, 4)), (try per_pos.dataConst())[0], 1e-6);
}
```

*(machine-verified snippet from docs/REFERENCE.md §4.15)*

`crossEntropyExt` adds the practical options with PyTorch parity:
`ignore_index` (padding contributes nothing — and when *every* position is
ignored the loss is 0, a deliberate, documented divergence from PyTorch's
NaN), `reduction` (`.mean`/`.sum`/`.none`), and `label_smoothing`.

One member of the family earns special attention:
`linearCrossEntropyExt(self, ctx, weight, labels, options)` fuses the final
projection *into* the loss. For an LLM, the logit matrix is `[rows, vocab]`
— often the largest tensor in the training step. The fused op computes the
logits once, keeps them on the backward record with the per-row softmax
statistics, and folds probability panels directly into the two input
gradients, reusing that saved buffer in place — "so the `[rows, classes]`
logit **gradient** is never materialized" (docs/REFERENCE.md §4.15).
Differentiable in both operands — the kind of memory-shape decision that
separates a library that can train language models on a CPU from one that
cannot ([Chapter 15](15-training-llms-on-cpu.md)).

The elementwise losses — `mseLoss`, `huberLoss`, `bceLoss`, `klDivLoss` —
follow the same option-struct pattern (docs/REFERENCE.md §4.15). One design
detail worth noticing: `bceLoss(.{ .from_logits = true })` uses the
numerically stable `max(x,0) − x·y + log1p(exp(−|x|))` form, and the
probability path clamps to `[1e-7, 1 − 1e-7]` (`bce_eps` in
`src/exec/loss.zig`) with gradient 0 outside the clamp — a documented
divergence from torch's enormous boundary gradients. The composed `nllLoss`
and `cosineSimilarity` require an exec scope under gradients (§5.3).

## 5.11 Selection, indexing, and the MoE doorway

### Choosing: argmax, topK, sort, routerTopK

One convention rules this corner of the library: **index outputs are
constant i64 tensors** — torch's index dtype, exact at any axis length —
and, being typed constants, they stay caller-owned even under a scope
(docs/REFERENCE.md §4.16).

```zig
test "argmax, topK, and routerTopK" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var logits = try fucina.Tensor(.{ .row, .expert }).fromSlice(&ctx, .{ 1, 4 }, &.{ 1, 3, 2, 0 });
    defer logits.deinit();

    var best = try logits.argmax(&ctx, .expert); // i64 indices, no grad
    defer best.deinit();
    try std.testing.expectEqualSlices(i64, &.{1}, try best.dataConst());

    var top = try logits.topK(&ctx, .expert, 2, .k);
    defer top.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 2 }, try top.values.dataConst());

    var selected: [2]usize = undefined;
    var weights: [2]f32 = undefined;
    try logits.routerTopK(&ctx, .expert, 2, .{}, &selected, &weights);
    try std.testing.expectEqual(@as(usize, 1), selected[0]);
    // normalize_selected renormalizes the top-k softmax mass to 1
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), weights[0], 1e-6);
}
```

*(machine-verified snippet from docs/REFERENCE.md §4.16)*

The gradient rules encode what each op *means*: `argmax` is no-grad by
design ("like sampling" — a hard choice has no useful derivative); `topK`'s
*values* are differentiable (the gradient scatters back through the saved
indices) while its indices are constants; `sort` is unstable and pins NaN
last regardless of direction (a documented divergence from `torch.sort`).
`routerTopK` is the mixture-of-experts router primitive: per row, softmax
over the *full* expert axis, pick the top k, optionally renormalize their
mass — filling caller-provided slices because its consumers are the
below-facade MoE entries we meet in a moment.

### Moving data: gather, scatter, and the functional updates

The indexing family (docs/REFERENCE.md §4.17) is where the "embedding
lookup" of every language model lives: `gather(ctx, tag, indices, out_tag)`
selects rows along a tag — token IDs in, embedding rows out. Its adjoint is
scatter-add: gradients from duplicate indices *accumulate*, which is exactly
what a token appearing twice in a batch should do to its embedding row.
`indexSelect` is the same op fed by a rank-1 **i64** index *tensor*, and
`takeAlongAxis` is the per-element variant — so `argmax`/`topK`/`argsort`
outputs feed them directly (any other index dtype is a compile error):
`x.takeAlongAxis(&ctx, .col, &order)` with
`order = x.argsort(&ctx, .col, false)` sorts each row — torch's
`gather(x, 1, x.argsort(1))` without the dim bookkeeping.

The rest of the family in one breath: `narrow`/`select`/`slice` (zero-copy
views on step-1 ranges — and `narrow` *aliases*: mutations of the source are
visible through it), `pad`, `concat`, `stack`, `flip`, `roll`,
`repeatAxis`; the functional updates `setSlice`/`setRows` (overwrite) and
`indexAdd` (accumulate); `scatterAdd`/`scatter` (per-element, torch
semantics — with a determinism refinement we will meet in §5.12);
`maskedSelect` (whose data-dependent "nothing matched" case gets the
dedicated `error.EmptySelection`, recoverable apart from real shape errors);
and `getRows` — a fused gather + dequantize on block-quantized weights, the
embedding path of every GGUF model
([Chapter 11](11-model-files-and-quantization.md)).

### The MoE doorway

Mixture-of-experts is conditional computation: a router picks `k` experts
per token and only those run — a model holds far more parameters than any
token's compute touches. The routed expert FFN runs *below* the tag facade,
directly on `ExecContext`, and is **inference-only** (docs/REFERENCE.md
§4.18):

```zig
pub fn moeExpertFfn(self: *ExecContext, x: *const Tensor,
    gate: *const MoeRhs, up: *const MoeRhs, down: *const MoeRhs,
    selected: []const usize, weights: []const f32,
    out_pe: usize, act: GatedOp, io: ?std.Io, profile: ?*MoeBatchProfile) !Tensor
```

*(signature from `src/exec.zig`; impl `src/exec/moe.zig`)*

It computes the route-weighted sum over the selected experts of
`down(act(gate(x), up(x)))` — §5.5's gated activation, routed. `MoeRhs` is a
tagged union stacking *all* experts of one projection into a single
quantized RHS (each expert a zero-copy sub-view); its arms cover the
resident quantized formats — `q4_k`, `q5_k`, `q6_k`, `q8_0`, `tq2_0`,
`ptqtp`, `q2_k`, `iq2_xxs`, `iq3_xxs`, `iq2_s`, `iq4_xs`, `q3_k` — plus a
`streamed` arm resolving expert blocks from disk through
`fucina.expert_store`: models larger than RAM decode through it
(`src/exec/moe.zig:77`; [Chapter 13](13-inference-tricks.md)).

> **Zig note** — `MoeRhs` is a `union(enum)`: a tagged union, Zig's sum
> type. One op fans out over many storage formats with a single `switch`,
> and the compiler checks exhaustiveness. Its `deinit` shows a lovely idiom:
> `switch (self.*) { .streamed => {}, inline else => |*value|
> value.deinit() }` — `inline else` stamps out one arm per remaining variant
> at comptime, each with the right concrete type (`src/exec/moe.zig`).

## 5.12 Determinism as a design stance

Most frameworks treat run-to-run reproducibility as a best-effort debug
mode. Fucina treats it as a contract — but a *precise* one, with two
distinct stories that must not be conflated.

### Story one: seed-driven ops are bitwise reproducible

The repo owns its RNG (`src/rng.zig`) instead of using `std.Random`,
because its "(seed → values) mappings are **checkpoint contracts**:
consumers store a seed and regenerate values instead of serializing them, so
none of these functions may ever change behavior" (docs/REFERENCE.md §6.8).
The core is counter-based random access — the i-th output of a stream in
O(1), no sequential state:

```zig
pub fn at(seed: u64, i: u64) u64 {
    var z = seed +% (i +% 1) *% gamma;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}
```

*(from `src/rng.zig:32`)*

> **Zig note** — `+%` and `*%` are Zig's *wrapping* arithmetic operators.
> Plain `+` on overflow is safety-checked illegal behavior (a panic in safe
> builds); hash mixing like this **wants** two's-complement wraparound, and
> Zig makes you say so in the operator itself.

Because every element is a pure function of `(seed, i)`, a parallel fill
splits any way you like and produces identical bits — the property is
test-pinned:

```zig
test "counter-based rng reproduces the sequential stream chunk by chunk" {
    const rng = fucina.rng;

    var state: u64 = 42;
    for (0..8) |i| {
        const sequential = rng.splitmix64(&state);
        try std.testing.expectEqual(sequential, rng.at(42, i)); // O(1) random access
    }

    var whole: [6]f32 = undefined;
    rng.gaussianFillAt(7, 0, &whole, 1.0);
    var parts: [6]f32 = undefined;
    rng.gaussianFillAt(7, 0, parts[0..2], 1.0);
    rng.gaussianFillAt(7, 2, parts[2..], 1.0); // any chunking, identical bits
    try std.testing.expectEqualSlices(f32, &whole, &parts);
}
```

*(machine-verified snippet from docs/REFERENCE.md §6.8)*

**Dropout is this contract turned into an op.** Dropout randomly zeroes
elements during training (a regularizer — the network cannot rely on any
single unit); most frameworks generate a mask tensor and keep it for the
backward pass. Fucina's stores nothing:

```zig
pub fn dropout(self: *const Self, ctx: *ExecContext, p: f32, seed: u64) !Self {
    if (p == 0) return self.withTags(ctx, tags);
    var value = try ctx.dropoutForward(self.asRawTensor(), p, seed);
    errdefer value.deinit();
    return finishOp(tags, ctx, value, self.requiresGrad(), DropoutBackward(tags), .{ ctx.allocator, self.grad_state, p, seed });
}
```

*(from `src/ag/tensor.zig:1244`)*

Element `i` survives iff the uniform draw of `rng.at(seed, i)` falls below
`1 − p`; "the mask is never stored: forward, backward, and checkpoint
recompute regenerate it from `(seed, index)`" (docs/REFERENCE.md §4.4). Look
at what the backward record captures: `(p, seed)` — two scalars, not a mask
tensor. The op is a pure function of `(input, p, seed)`, bitwise stable
under any parallel chunking. Two caller-side rules follow: pass a *fresh*
seed per call (reusing a seed reuses the mask), and eval mode is simply not
calling dropout.

The same contract powers APOLLO's regenerated low-rank projections and
evolution strategies' regenerated perturbations
([Chapter 9](09-training-without-gradients.md)) — gigabytes of noise that
never touch a checkpoint file. One precision matters:
`gaussianFillAtFast`, the vectorized fill ES uses, is a **distinct**
(seed → values) mapping from `gaussianFillAt` — values agree to a few ulps
but are *not bitwise equal* — so it is a separate checkpoint contract,
equally chunking-invariant (docs/REFERENCE.md §6.8). Swapping one for the
other under an existing seed is a silent behavior change.

### Story two: threaded kernels say exactly what they promise

A reduction split across threads *can* legitimately differ in the last bits
from its serial order. Fucina does not paper over this with a blanket
claim; it draws the line per kernel class (docs/REFERENCE.md §9.4):

> "Parallel splits are deterministic: tasks own disjoint output ranges, so
> the threaded result is bit-identical to the serial path for elementwise,
> conv, pool, and Winograd kernels (reductions and GEMM state their
> reassociation tolerance instead)."

So: where each task owns a disjoint slice of the output — elementwise ops,
convolutions, pooling, Winograd tiles — threading changes *nothing*, ever.
Where tasks must combine partial sums — reductions and GEMM — the result is
tolerance-equivalent, not bit-identical, and the cross-backend parity suite
states the tolerance explicitly: sums and dots agree within `1e-6·n`
(scaling with the accumulation count), matmuls within `1e-5·k`
(docs/REFERENCE.md §9.3). "No blanket bitwise-reproducibility claim is made
for every parallel reduction at every thread count; where a kernel
guarantees serial/parallel parity, the guarantee is pinned by its tests"
(docs/REFERENCE.md §6.8).

Keep the two stories separate in your head: *seed-driven randomness* is
bitwise reproducible across runs, thread counts, and chunk decompositions;
*threaded numeric kernels* are bitwise reproducible exactly where the docs
say so, and tolerance-equivalent where summation order is at stake.

### Pinning what others leave unspecified

The same temperament shows up in op semantics. `scatter` resolves duplicate
indices "deterministically to the LAST row-major write (torch leaves the
order unspecified; this pins it)" (docs/REFERENCE.md §4.17); `cumsum` is
serial-per-row by default so sequences are exact (§5.7); `sort` documents
its NaN placement; integer tensors get no `div` at all — torch silently
promotes to float, Fucina makes you choose `divTrunc` or `divFloor`
(docs/REFERENCE.md §4.19). The pattern: match PyTorch semantics wherever
reasonable, and where PyTorch is unspecified or surprising, *pick a
behavior, document it as a divergence, and pin it with a test*.

## 5.13 Extending the library: elemental ops

The kernel enums are closed (§5.5) — so when your model needs an activation
Fucina never heard of, you write two scalar functions:

```zig
const Square = struct {
    pub fn forward(x: f32, extra: void) f32 {
        _ = extra;
        return x * x;
    }
    // Returns the propagated dL/dx, NOT the local dy/dx.
    pub fn backward(x: f32, y: f32, grad_y: f32, extra: void) f32 {
        _ = y;
        _ = extra;
        return 2 * x * grad_y;
    }
};
```

*(machine-verified helper from docs/REFERENCE.md §4.4)*

and lift them with `elementalUnary`:

```zig
test "elementalUnary" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, -2, 3 });
    defer x.deinit();
    var y = try x.elementalUnary(&ctx, Square, {});
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 9 }, try y.dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, -4, 6 }, try gx.dataConst());
}
```

*(machine-verified snippet from docs/REFERENCE.md §4.4)*

"The user writes scalar math only; the adapter owns buffer plumbing,
strided-input materialization, tag-driven broadcasting, broadcast-gradient
sum-reduction, `needs_grad` pruning, and the worker-team chunking of the
scalar loops" — chunking that is "bitwise thread-count-neutral: disjoint
pure writes" (docs/REFERENCE.md §4.4) — §5.12's story two holding for your
code too. `elementalBinary` does the same for two-operand ops, with the
full §5.4 broadcast rule and per-operand `backwardA`/`backwardB`.

> **Zig note** — `Square` is passed as `comptime Op: type`: a *type* used as
> a compile-time module of functions. The adapter calls `Op.forward` in its
> inner loop, and because the type is comptime-known, the call inlines —
> your scalar function compiles into the loop body, not behind a function
> pointer. The mechanics fit in four lines of course code (compile-checked):
>
> ```zig
> // Course code — the shape of the elemental tier, not repo code.
> fn mapUnary(comptime Op: type, out: []f32, in: []const f32) void {
>     for (out, in) |*o, x| o.* = Op.forward(x);
> }
> ```

Two contract details to respect: `backward` returns the *propagated*
`dL/dx` (multiply by `grad_y` yourself — the snippet's comment is there
because everyone forgets once), and `extra` is captured **by value** in the
backward node, so any pointers inside it must outlive the backward pass.
Verify a real one with `fucina.gradcheck` ([Chapter 7](07-autograd.md)) —
gradients are checked against finite differences in this codebase, never
trusted. For ops that are not elementwise — a custom contraction, a fused
block — the heavier escape hatch is `fucina.customVjp`, also Chapter 7
material.

## What you now know

- Fucina is **eager**: an op call validates, runs a kernel, and returns an
  owned tensor — no graph object, no deferred execution, anywhere.
- `ExecContext` is the whole runtime in one self-referential struct: init it
  in place, never move it, drive it from one thread; it owns the pool, the
  backend, the lazy worker team, and the scope stack.
- Every op shares one contract — comptime tags for axis choice, recoverable
  errors for size mismatches, caller-owned results, borrowed operands —
  implemented once in the `finishOp`/`finishNoGrad` tails; the shape is
  *validate once at the facade, dispatch to small unchecked kernels* that
  classify layout once and never allocate.
- Broadcasting, reduction, and contraction are all *tag-directed*: no
  transpose dance around `dot`/`einsum`, and gradients into broadcast
  operands sum-reduce back automatically.
- The op families are the ML vocabulary: activations (a zoo with history),
  gated FFNs, masks, softmax (a decade of attention tricks in its options),
  RMSNorm/LayerNorm, RoPE, grouped attention, convolutions, fused losses,
  selection, and gather/scatter — each with its differentiability rules
  stated, including the inference-only entries (packed-RHS GEMMs,
  `moeExpertFfn`).
- Ownership has two disciplines: deinit-ASAP (and `ctx.replace`) for
  inference, exec scopes for training — and scopes are *not* an inference
  optimization (measured 2 vs 32 live buffers on a 32-op chain).
- Determinism is two distinct promises: seed-driven ops (counter-based RNG,
  dropout, ES noise) are bitwise reproducible across runs, threads, and
  chunkings; threaded numeric kernels are bit-identical only where tasks own
  disjoint output (elementwise/conv/pool/Winograd) and tolerance-equivalent
  where they reduce (reductions, GEMM). `gaussianFillAtFast` is its own
  seed→values contract, a few ulps from `gaussianFillAt`, never bitwise.
- Extensibility lives at the scalar level: `elementalUnary`/`elementalBinary`
  lift your two scalar functions into differentiable, broadcast-aware,
  parallel tensor ops.

## Explore the source

- `src/exec.zig` — `ExecContext`: the facade, `LayoutClass`, the scope
  machinery, and the ~300-entry raw op surface with its naming grammar
  (`*Rank`, `*AxisRank`, `*Typed`, `*Backward*`).
- `src/ag/tensor.zig` — the tagged op facade: read `add`, then `scale`, then
  `finishOp`/`finishNoGrad`, and you have read the whole library's skeleton.
- `src/tagged.zig` — the lowering tier: `pointwise` is the 25-line
  crystallization of "validate once, view, dispatch".
- `src/exec/softmax.zig`, `src/exec/loss.zig`, `src/exec/topk.zig`,
  `src/exec/moe.zig` — leaf modules (not public API, but the best reading on
  how each family really works).
- `src/rng.zig` — the counter-based RNG contract, six lines at its core.
- `docs/REFERENCE.md` §4 and §6 — the machine-verified catalogue this
  chapter sampled; every snippet above runs in CI.
- `docs/MEMORY-MODEL.md` — the design record behind ownership, the pool,
  and the scope adjudications.

## Exercises

1. **Softmax sanity.** Build a `Tensor(.{ .row, .col })` of arbitrary
   values, softmax over `.col`, and verify each row sums to 1. Then
   reimplement relu three ways — `x.relu(&ctx)`, `compare` + `maskedFill`,
   and `x.maximum(&ctx, &zeros)` — and check all three agree exactly.
2. **Chunk-invariance, extended.** Split a 12-element `gaussianFillAt` into
   three uneven chunks and verify bitwise equality with the whole fill. Do
   the same for `gaussianFillAtFast` against itself — then confirm the two
   functions do *not* agree bitwise on the same seed.
3. **Your own activation.** Implement ELU (`x >= 0 ? x : exp(x) − 1`) as an
   elemental `Op` and lift it with `elementalUnary`. Check your values
   against the built-in `x.unary(&ctx, .elu)` (alpha = 1); validate your
   `backward` with `fucina.gradcheck` once you reach
   [Chapter 7](07-autograd.md).
4. **One equation, three spellings.** Compute attention-style scores
   `q[.seq, .d] × k[.src, .d] → [.seq, .src]` with (a) `dot` over `.d`,
   (b) `einsum` with `out_tags = .{ .seq, .src }`, and (c) explicit `matmul`
   with `.trans_b`. Verify all three agree, and explain why (b) needed no
   transposed operand.
5. **See the pool breathe.** Run 16 same-shaped `scale` ops in deinit-ASAP
   style, collecting `dataConst().ptr` of each result before releasing it;
   count the distinct addresses. Repeat under an open exec scope. You should
   reproduce the shape of the "2 vs 32 distinct buffers" measurement from
   docs/MEMORY-MODEL.md §5 — then explain both numbers.

[Previous: Axes with names](04-axes-with-names.md) · [Next: Going fast on CPUs](06-going-fast-on-cpus.md)
