# Chapter 08 — Training: making the machine learn

*Part III — Learning*

[Chapter 7](07-autograd.md) ended with a superpower: call `backward()` on a
loss and every parameter's gradient appears, computed by the graph hidden in
the live tensors. But a gradient is only a *direction*. Nothing has learned
anything yet. This chapter builds the machinery that turns gradients into
learning — the training step as a fixed ritual, loss functions with their
sharp edges, optimizers from a three-line SGD to the research frontier,
learning-rate schedules, gradient clipping and accumulation, checkpoints that
resume **bit-exactly**, and mixed precision that halves memory without
corrupting the math. At the end, the payoff: `examples/spirals/main.zig`, a
complete trainable model in under 500 lines, which you will run and watch
learn.

Everything here lives in one file pair you can actually read:
`src/optim.zig` (the optimizers, schedules, clipping, persistence — about
2,900 lines) and `docs/TRAINING.md` (the manual this chapter follows and
quotes). No trainer framework, no callback hierarchy, no `Trainer` god
object. A training loop in Fucina is a `for` loop you write yourself, and
every line of it does something you can point at.

## 8.1 Anatomy of a training step

Here is a complete training step, verbatim from `docs/TRAINING.md` §1 — this
snippet is the spine of the whole chapter, and each of its stages gets a
section below:

```zig
const fucina = @import("fucina");
const optim = fucina.optim;

var opt = optim.AdamW.init(allocator, .{ .lr = 1e-3, .weight_decay = 0.01 });
defer opt.deinit();
try opt.addParam(&w1);                            // params must outlive opt
try opt.addParam(&b1);

var sched = optim.LrSchedule.init(allocator);
defer sched.deinit();
try sched.attach(&opt.config.lr);

for (0..total_steps) |step_i| {
    sched.apply(optim.warmupCosineFactor(step_i, total_steps, warmup, 0.1));
    const scope = ctx.openExecScope();            // scope owns the step's intermediates
    defer ctx.closeExecScope(scope);
    const h = try x.dot(&ctx, &w1, .in);          // no keeps, no defers
    const a = try h.add(&ctx, &b1);
    const z = try a.tanh(&ctx);
    const loss = try z.crossEntropy(&ctx, .class, labels);
    try loss.backward(&ctx);
    _ = try opt.clipGradNorm(&ctx, 1.0);          // after backward, before step
    try opt.step(&ctx);
    opt.zeroGrad();
}
```

Read it as six stages, in an order that is not negotiable:

1. **Forward** — ordinary op calls, exactly the inference code from
   [Chapter 5](05-the-operation-library.md). The eager graph builds itself
   as a side effect ([Chapter 7](07-autograd.md)).
2. **Loss** — one more op, reducing the batch to a scalar (§8.3).
3. **`backward()`** — walk the graph, *add* each parameter's gradient into
   its persisted accumulator (that word "add" matters; it is what makes
   gradient accumulation free in §8.8).
4. **`clipGradNorm`** — optionally rescale the gradients if their global
   norm exceeds a bound (§8.7). It must see the *complete* gradients, hence
   after backward; it must run before they are consumed, hence before step.
5. **`step()`** — the optimizer reads the gradients (non-destructively) and
   updates the parameters (§8.4–8.6).
6. **`zeroGrad()`** — clear the accumulators, ending the step.

`docs/REFERENCE.md` §11 calls this the canonical step order:
"`backward` → `clipGradNorm` → `step` → `zeroGrad`", and the repo pins it
with tests. Everything else in this chapter — schedules, groups, mixed
precision, checkpointing — is machinery that makes one of these six stages
correct, fast, or resumable.

> **ML note** — Why is `zeroGrad` a separate call instead of something
> `step()` does implicitly? Because *not* calling it is a feature: gradients
> that survive across steps are exactly how you train on batches larger than
> memory (gradient accumulation, §8.8). PyTorch made the same choice for the
> same reason. The rule of thumb: every knob in this API corresponds to a
> real degree of freedom in training, not to framework ceremony.

## 8.2 Training lifetime is not inference lifetime

Chapter 7 introduced exec scopes; here is why training *needs* them. In
inference, the discipline you learned in
[Chapter 3](03-tensors-from-scratch.md) is: deinit every intermediate the
moment its consumer has run. Released buffers go straight back to the pool;
memory stays flat. `docs/TRAINING.md` §2 states the training rule that
replaces it:

> **Training rule:** every tensor on the path from the parameters to the
> loss must stay alive until `backward()` returns.

The reason is an ownership asymmetry inside each op result. Its **value**
(the `RawTensor`) is refcounted — backward functions clone views of the
operand values they need, so releasing a value early never dangles data. Its
**GradState** — the autograd graph node — is not. Quoting TRAINING.md §2:
it is "**single-owner, not refcounted**: the tensor owns it,
`tensor.deinit()` destroys it unconditionally, and the consumers' backward
functions hold **raw `*GradState` pointers** to it". Deinit an intermediate
before backward and the backward pass walks a dangling node — undefined
behavior, not a catchable error. The design is deliberate: "no atomic
refcount traffic on the eager hot path, no ownership cycles, and inference
(where `grad_state == null`) pays nothing. The price is a rule"
(docs/TRAINING.md).

The rule would make training code miserable — every intermediate carefully
kept, then carefully released after backward — if the library asked you to
follow it by hand. It doesn't. **Exec scopes** invert the discipline: while
a scope is open, every op result is owned by the scope, its `deinit` is a
safe no-op, and closing the scope releases everything, newest first:

```zig
const scope = ctx.openExecScope();
defer ctx.closeExecScope(scope);     // releases everything, newest first
const logits = try forwardLogits(&ctx, &model, &x);   // zero ceremony inside
```

The consequence is the property this library keeps advertising: *training
forward code looks like inference*. You write the forward once, in the plain
deinit-ASAP style; run it bare and it infers; wrap a scope around it and the
graph survives to `backward()`. Two boundaries remain yours: tensors you
create explicitly (`variable`, `constant`, `fromSlice`) and gradients you
fetch stay caller-owned, and you must never *use* a scope-owned borrow after
the scope closes — that is the one hazard the `scope_owned` flag cannot
remove (docs/TRAINING.md §12).

Two roads not taken, both recorded in TRAINING.md §2: an explicit per-tensor
owner (`Tape`) "was prototyped early on and removed unused", and scopes
deliberately do **not** replace deinit-ASAP inside the inference engines —
"that option was evaluated and rejected; see the note at the end of
MEMORY-MODEL §5". When a design decision in this repo loses, the losing
option gets a tombstone you can read.

> **Zig note** — A scope is arena semantics grafted onto an eager runtime:
> the owner (the `ExecContext`, already threaded through every op call)
> adopts allocations and frees them in bulk at a boundary. If you know Zig's
> `std.heap.ArenaAllocator`, the shape is familiar; the twist is that
> adoption happens inside the op tails (`finishOp` in `src/ag/tensor.zig`),
> so user code opts in by opening a scope, not by passing a different
> allocator. And because the scope owns the prefix of results the moment
> each op returns, a failed op mid-forward leaks nothing — model code needs
> no `errdefer` chains at all.

## 8.3 Losses: cross-entropy and its knobs

A loss is just one more differentiable op, and for classification the op is
cross-entropy. The plain form is what the training-step snippet used —
`x.crossEntropy(ctx, .class, labels)`, mean cross-entropy over rows, with
the class axis named by tag ([Chapter 4](04-axes-with-names.md)) rather than
by position. The extended form `x.crossEntropyExt(ctx, .class, labels,
options)` adds the PyTorch-parity knobs (`exec.CrossEntropyOptions`,
`src/exec/loss.zig`; the summary below follows `docs/TRAINING.md` §5):

- **`ignore_index`** — a sentinel label; those positions contribute zero
  loss *and zero gradient*, and are excluded from the `.mean` denominator.
  This is what makes supervised fine-tuning work: prompt tokens get the
  sentinel, response tokens get real labels, and the model is only graded
  on what it should learn to say ([Chapter 15](15-training-llms-on-cpu.md)).
  One documented divergence from PyTorch: when *every* position is ignored
  the loss is 0, not NaN.
- **`reduction = .mean | .sum | .none`** — `.none` returns per-position
  losses with the class tag removed. `.sum` looks like a convenience and is
  actually load-bearing: it is the correct arm for gradient accumulation
  with uneven token counts (§8.8).
- **`label_smoothing`** in [0, 1) — target = (1−ε)·onehot + (ε/K)·uniform,
  PyTorch semantics.

Determinism gets a sentence of its own, because it underwrites §8.9: the row
kernels are SIMD and parallel over rows, but "each row writes its loss to a
per-row buffer and the `.mean`/`.sum` reduction is ONE serial sum in row
order …, so the loss is bitwise stable across thread counts"
(docs/TRAINING.md §5). Speed and reproducibility, not speed instead of it.
As a dated, machine-specific snapshot (M1 Max, ReleaseFast,
`zig build bench-ce`; docs/TRAINING.md §5): CE forward at 1024×151936 went
354.5 ms → ~18 ms and backward 426 ms → ~22 ms (~19–20×) when the row
kernels went SIMD and parallel over rows — measured, not asserted, and
specific to that machine.

For language models there is one more trick worth knowing now, because it
shows what "the graph is just the values" makes possible. The final
projection of an LLM multiplies a `[rows, hidden]` activation by a
`[vocab, hidden]` weight to produce `[rows, vocab]` logits — at a 151936-word
vocabulary, the *gradient* of that logit tensor is hundreds of megabytes.
`x.linearCrossEntropyExt(ctx, &w, labels, options)` fuses projection and
loss into ONE differentiable op: the logits never enter the graph, and the
backward overwrites them in place with the logit gradient before the two
gradient GEMMs, "so the full `[rows, classes]` gradient never costs a second
buffer (−622 MB peak and ~4% faster than the composed dot + crossEntropyExt
backward at 1024x151936x1024 on M1)" (docs/TRAINING.md §5 — again a dated
M1 snapshot, not a promise).

## 8.4 Optimizers from scratch: SGD, then momentum

Strip everything away and an optimizer is one line. The following is course
code (not from the repo; compile-checked with `zig test`):

```zig
// Course code — NOT from the Fucina repo.
/// Plain gradient descent: walk downhill, step size lr.
fn sgdStep(params: []f32, grads: []const f32, lr: f32) void {
    for (params, grads) |*p, g| p.* -= lr * g;
}
```

That is genuinely all of stochastic gradient descent: move each parameter a
small step against its gradient. Paired with Chapter 7's `backward()`, this
one-liner already trains the spirals MLP to convergence — deep learning's
dirty secret is how far the simplest possible update goes.

> **Zig note** — `for (params, grads) |*p, g|` iterates two slices in
> lockstep (they must have equal length — checked in safe builds), capturing
> the first element *by pointer* (`*p`, so we can mutate it) and the second
> by value. This multi-slice `for` with mixed captures is the idiomatic Zig
> replacement for an indexed loop, and the real optimizer kernels in
> `src/optim.zig` use it with four slices at once.

The first upgrade is **momentum**: keep a velocity buffer that remembers the
recent direction of travel, and step along the smoothed direction instead of
the raw gradient. Course code again:

```zig
// Course code — NOT from the Fucina repo.
fn momentumStep(params: []f32, grads: []const f32, velocity: []f32, lr: f32, momentum: f32) void {
    for (params, grads, velocity) |*p, g, *v| {
        v.* = momentum * v.* + g; // accumulate direction
        p.* -= lr * v.*; // step along the smoothed direction
    }
}
```

> **ML note** — Why does this help? Gradients from small batches are noisy,
> and loss surfaces have narrow valleys: raw SGD zig-zags across the valley
> walls while making slow progress along the floor. The velocity buffer is
> an exponential moving average of gradients — perpendicular noise cancels,
> the consistent along-the-valley component accumulates. The physical
> analogy in the name is exact: a heavy ball rolling downhill doesn't
> reverse direction at every pebble.

Now meet the real thing. `optim.SGD` (`src/optim.zig`) is not "an SGD-like
optimizer"; it is a port of `torch.optim.SGD`, quirks included. Its config
struct is the whole hyperparameter story — one field per knob, defaults
visible in the source (`src/optim.zig:2038`):

```zig
pub const SgdConfig = struct {
    lr: f32 = 1e-3,
    /// 0 disables the momentum buffer entirely (no state RAM).
    momentum: f32 = 0,
    dampening: f32 = 0,
    /// COUPLED L2 (g += wd*p), like PyTorch SGD — not AdamW-style decoupled.
    weight_decay: f32 = 0,
    /// Requires momentum > 0 and dampening == 0 (PyTorch constructor rule).
    nesterov: bool = false,
    /// Storage dtype of the momentum buffer; step math stays f32. ...
    state_dtype: StateDType = .f32,
};
```

Two quirks are worth pausing on, because each teaches something:

**The first-step clone.** PyTorch initializes the momentum buffer not to
zeros but to a clone of the first (decayed) gradient, and so does this port
(the comment at `src/optim.zig:2059-2067` spells it out). Getting this wrong
would still converge — it would just diverge *numerically* from the
reference, and this repo's bar for an optimizer is a golden parity test
against the actual reference implementation ("PyTorch 2.12 / Keller
Jordan's muon.py / apollo_torch", docs/TRAINING.md §3; the goldens live in
`src/optim_tests.zig`). An optimizer that is "basically Adam" is a bug you
cannot see until your training run mysteriously underperforms a paper.

**The constructor panic.** From `src/optim.zig:2069-2076`:

```zig
pub fn init(allocator: Allocator, config: SgdConfig) SGD {
    // PyTorch constructor rule, enforced in every build mode (a debug
    // assert would vanish exactly where training runs: ReleaseFast).
    if (config.nesterov and (config.momentum == 0 or config.dampening != 0)) {
        @panic("SGD: nesterov requires momentum > 0 and dampening == 0");
    }
    return .{ .allocator = allocator, .config = config };
}
```

> **Zig note** — Zig gives you three tools for "this must not happen", and
> this snippet is a masterclass in choosing between them. An **error union**
> (`!SGD`) is for conditions the caller can meaningfully handle — but a
> nonsensical hyperparameter combination is a bug in the program, not a
> runtime circumstance. A **`std.debug.assert`** compiles away in
> ReleaseFast — which is exactly the build mode training runs in, so the
> check would vanish precisely where it matters. **`@panic`** survives every
> build mode and names the invariant in the crash message. Config
> validation at a constructor is the `@panic` case.

## 8.5 AdamW: the workhorse, ported exactly

SGD treats every parameter alike. Adam gives each parameter its own
adaptive step size by tracking two exponential moving averages: `m`, the
mean of the gradient (momentum), and `v`, the mean of its square (a
per-element estimate of gradient scale). The update divides by `sqrt(v)`,
so parameters with consistently large gradients take small steps and
vice versa. AdamW is Adam with one change that took the field years to get
right: **decoupled weight decay**. The difference is literally one line —
the module doc of `src/optim.zig:5-8` states both variants:

> - `Adam` — PyTorch `torch.optim.Adam` single-tensor path (coupled L2
>   weight decay: `g += weight_decay * p` before moment updates).
> - `AdamW` — PyTorch `torch.optim.AdamW` single-tensor path (decoupled
>   decay applied to the parameter BEFORE the Adam step; `denom =
>   sqrt(v)/sqrt(1-b2^t) + eps`).

> **ML note** — Coupled decay adds `wd·p` to the gradient, which then flows
> through Adam's adaptive rescaling — so the *effective* regularization
> strength varies per parameter with its gradient history, which is almost
> never what you meant. Decoupled decay shrinks the parameter directly,
> uniformly, outside the adaptive machinery. That one-line difference is
> the "W" in AdamW and the subject of an entire paper (Loshchilov &
> Hutter, 2017).

Here is the whole per-element AdamW update as it actually ships, verbatim
from `src/optim.zig:756-767` — twelve lines:

```zig
fn runScalar(c: @This(), start: usize, end: usize) void {
    for (c.p[start..end], c.g[start..end], c.m[start..end], c.v[start..end]) |*pi, gi, *mi, *vi| {
        const decayed = pi.* * c.s.keep;
        const m0 = stateLoad(md, mi);
        const v0 = stateLoad(vd, vi);
        const m1 = m0 + c.s.one_minus_b1 * (gi - m0);
        const v1 = c.s.beta2 * v0 + c.s.one_minus_b2 * gi * gi;
        stateStore(md, mi, m1);
        stateStore(vd, vi, v1);
        pi.* = decayed - c.s.step_size * (m1 / (@sqrt(v1) / c.s.bc2s + c.s.eps));
    }
}
```

Walk it: `decayed` is the decoupled decay (`keep = 1 − lr·wd`); `m1`/`v1`
are the two EMAs (the `stateLoad`/`stateStore` indirection is mixed
precision for the *optimizer state*, §8.10); the last line is the
bias-corrected step. The scalars arrive pre-computed, and where they come
from matters (`src/optim.zig:776-799`, trimmed):

```zig
fn adamwUpdate(ctx: *ExecContext, config: AdamWConfig, p: []f32, g: []const f32, m: StateBuf, v: StateBuf, step_count: u64) void {
    const t: f64 = @floatFromInt(step_count);
    const bc1 = 1 - std.math.pow(f64, config.beta1, t);
    const bc2_sqrt = @sqrt(1 - std.math.pow(f64, config.beta2, t));
    const s = AdamWScalars{
        .keep = if (config.weight_decay != 0) @floatCast(1.0 - @as(f64, config.lr) * @as(f64, config.weight_decay)) else 1,
        // ... one_minus_b1, one_minus_b2, step_size = lr / bc1, bc2s, eps ...
    };
    switch (m) {
        .f32 => |ms| switch (v) {
            .f32 => |vs| adamwRun(.f32, .f32, ctx, s, p, g, ms, vs),
            .bf16 => |vs| adamwRun(.f32, .bf16, ctx, s, p, g, ms, vs),
        },
        .bf16 => |ms| switch (v) { /* mirror arms */ },
    }
}
```

Two things to notice. First, the bias corrections `bc1`/`bc2_sqrt` are
computed **once per step, in f64**, then rounded to f32 — matching, per
`docs/REFERENCE.md` §11, "torch's Python-float scalars to within a few f32
ulps". Early in training, `1 − β₂ᵗ` is a tiny number computed by subtracting
two nearly-equal ones; doing that in f32 per element would both waste time
and lose precision. Second, the nested `switch` on the two state-dtype
unions monomorphizes the kernel: each of the four (m, v) dtype combinations
gets its own compiled loop, and the dispatch happens once per tensor, not
once per element.

> **ML note** — *Bias correction*, demystified: an EMA initialized at zero
> is biased toward zero for its first ~1/(1−β) steps (`v` starts at 0 and
> only slowly fills up). Dividing by `1 − βᵗ` rescales the estimate to be
> unbiased from step one. Without it, early steps divide by an
> underestimated `sqrt(v)` and blow up — which is why the correction is in
> the algorithm and not a tuning nicety.

If you want to *feel* the algorithm before trusting the port, here is a
from-scratch version (course code, compile-checked; it converges on a toy
quadratic and reproduces the decay-pulls-below-the-minimum effect):

```zig
// Course code — NOT from the Fucina repo.
const CourseAdamW = struct {
    lr: f32 = 1e-3,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-8,
    weight_decay: f32 = 0.01,
    t: u64 = 0, // step count, for bias correction

    fn step(self: *CourseAdamW, p: []f32, g: []const f32, m: []f32, v: []f32) void {
        self.t += 1;
        const t: f64 = @floatFromInt(self.t);
        const bc1: f32 = @floatCast(1.0 - std.math.pow(f64, self.beta1, t));
        const bc2_sqrt: f32 = @floatCast(@sqrt(1.0 - std.math.pow(f64, self.beta2, t)));
        const keep = 1.0 - self.lr * self.weight_decay;
        for (p, g, m, v) |*pi, gi, *mi, *vi| {
            pi.* *= keep; // decoupled decay: shrink the param, not the gradient
            mi.* += (1.0 - self.beta1) * (gi - mi.*); // EMA of the gradient
            vi.* = self.beta2 * vi.* + (1.0 - self.beta2) * gi * gi; // EMA of its square
            pi.* -= (self.lr / bc1) * (mi.* / (@sqrt(vi.*) / bc2_sqrt + self.eps));
        }
    }
};
```

The real `AdamWConfig` (`src/optim.zig:526`) reads like the course struct
plus the two state-dtype knobs: `lr: f32 = 1e-3, beta1: f32 = 0.9, beta2:
f32 = 0.999, eps: f32 = 1e-8, weight_decay: f32 = 0.01, state_dtype:
StateDType = .f32, second_moment_dtype: StateDType = .f32`.

One more piece of the shared surface deserves a look: how a parameter gets
*into* an optimizer. `addParam(&w)` accepts any variable tensor via
`anytype` and builds a `Param` handle (`src/optim.zig:403`): a refcounted
view of the storage, a raw `*GradState` pointer, and a rows/cols flattening
for the matrix-aware optimizers. Two ownership rules follow directly and
are documented in the module header (`src/optim.zig:22-28`): parameters
must **outlive** the optimizer, and each variable belongs to exactly one
optimizer — duplicates within one instance are rejected
(`error.DuplicateParam`), while the same tensor in two *different*
instances is per-instance undetectable and silently double-steps (§8.7
shows how `OptimizerSet` closes that gap).

Finally, determinism — the property §8.9 will cash in. The update loops
parallelize across the worker pool, and the comment at
`src/optim.zig:85-91` explains why that costs nothing in reproducibility:

```zig
/// Elementwise update loops chunk across the worker pool above this length.
/// Every parallel loop here is an element-independent map (no reductions), so
/// the results are bitwise identical to the serial path for any thread count
/// — goldens and bit-exact resume are unaffected.
// … (the comment goes on to cover reductions; paraphrased below) …
const parallel_map_min_len: usize = 1 << 17;
```

An element-independent map has no cross-thread arithmetic, so the chunking
is invisible to the result. Reductions (the norms used by clipping) go
through `sumSquares` — a fixed chunk grid with a pinned combine order —
so they are equally thread-count-invariant.

## 8.6 The frontier: Muon and APOLLO

AdamW treats a weight matrix as a bag of independent scalars. The two
frontier optimizers in the tree stop doing that, in opposite directions:
Muon exploits matrix structure to take *better* steps; APOLLO exploits it
to take *cheaper* ones. Both are faithful ports pinned by goldens against
their reference implementations (Keller Jordan's `muon.py` and
`apollo_torch`; docs/TRAINING.md §3, `src/optim_tests.zig`).

### Muon: orthogonalized momentum

Muon's idea: take the momentum-averaged gradient of a weight *matrix*,
replace it with the nearest orthogonal matrix (approximately — via
Newton-Schulz iteration), and step along that. The update has balanced
singular values, so no direction in parameter space dominates. The
orthogonalization kernel is 30 lines and worth reading whole — verbatim
from `src/optim.zig:1386-1416`:

```zig
pub fn newtonSchulz5(ctx: *ExecContext, u: *const RawTensor, steps: u32) !RawTensor {
    const rows = u.shape.at(0);
    const cols = u.shape.at(1);
    const transposed = rows > cols;
    var x = if (transposed) try transpose2D(ctx, u) else try ctx.materialize(u);
    errdefer x.deinit();

    const sumsq = try sumSquares(ctx, x.dataConst());
    const inv_norm: f32 = @floatCast(1.0 / (@sqrt(sumsq) + 1e-7));
    for (x.data()) |*value| value.* *= inv_norm;

    for (0..steps) |_| {
        var gram = try ctx.matmulTransB(&x, &x);
        defer gram.deinit();
        var quad = try ctx.matmul2D(&gram, &gram);
        defer quad.deinit();
        for (quad.data(), gram.dataConst()) |*qi, gi| qi.* = ns_coeff_b * gi + ns_coeff_c * qi.*;
        var bx = try ctx.matmul2D(&quad, &x);
        errdefer bx.deinit();
        for (bx.data(), x.dataConst()) |*oi, xi| oi.* = ns_coeff_a * xi + oi.*;
        x.deinit();
        x = bx;
    }

    if (transposed) {
        const out = try transpose2D(ctx, &x);
        x.deinit();
        return out;
    }
    return x;
}
```

Normalize the matrix, then iterate a fixed odd polynomial in `X·Xᵀ` five
times: each pass pushes the singular values toward 1 while leaving the
singular *vectors* alone. The transpose trick keeps the Gram matrix at the
smaller of the two dimensions. And note the doc comment right above it
(`src/optim.zig:1384-1385`): "The result approximates U*V^T with singular
values in roughly (0.5, 1.5) — by design, not a bug." Approximate
orthogonality is all the optimizer needs, and exact SVD would cost far
more.

`MuonConfig` (`src/optim.zig:1074`; defaults per `docs/REFERENCE.md` §11):
`lr: f32 = 0.02, momentum: f32 = 0.95, nesterov: bool = true, ns_steps:
u32 = 5, weight_decay: f32 = 0, scale: MuonScale = .spectral, state_dtype:
StateDType = .f32, fallback: AdamWConfig = .{ .lr = 3e-4, .beta1 = 0.9,
.beta2 = 0.95, .eps = 1e-10, .weight_decay = 0 }`. Two details:

- `scale` selects the update-scale convention: Keller's spectral
  `sqrt(max(1, rows/cols))` or Moonlight's RMS-matching
  `0.2*sqrt(max(rows, cols))` (`MuonScale = enum { spectral,
  match_rms_adamw }`).
- That `fallback` field is a whole embedded AdamW, because Muon's math only
  makes sense for weight *matrices*. Biases and norms (0D/1D) auto-route
  to the fallback; embeddings and classifier heads are 2D but must not be
  orthogonalized, so *you* route them with `addFallbackParam` — forgetting
  is one of the checklist pitfalls: "silently wrong algorithm for those
  weights" (docs/TRAINING.md §12).

Muon's cost is real and honestly documented: the Newton-Schulz GEMMs are
"~395 GFLOP per block in f32 at 5 iterations" (docs/TRAINING.md §11), which
is why in the bench table below it sits two orders of magnitude above
AdamW. That is the algorithm, not the implementation.

### APOLLO: moments in a compressed space

APOLLO (arXiv 2412.05270) attacks the other end: AdamW's state costs
8 bytes per parameter, which at LLM scale is more than the model. APOLLO
projects each gradient matrix through a *random low-rank projection*,
keeps the Adam moments in that compressed space, and derives per-channel
(or per-tensor) scaling factors for a scaled-SGD update. `ApolloConfig`
(`src/optim.zig:1432`; defaults per `docs/REFERENCE.md` §11): `lr: f32 =
0.01, beta1: f32 = 0.9, beta2: f32 = 0.999, eps: f32 = 1e-6, weight_decay:
f32 = 0, rank: usize = 128, update_proj_gap: u64 = 200, scale: f32 = 1.0,
scale_type: ApolloScaleType = .channel, correct_bias: bool = true, …,
seed: u64 = 0`, plus `ApolloConfig.mini()` for the rank-1 variant.

The detail with the biggest lesson in it: the projection matrices are
**regenerated, not stored**. From TRAINING.md §8: "P is a deterministic
function of (seed, step / update_proj_gap) computed by the repo-owned
splitmix64 + Box-Muller generator … deliberately not `std.Random`, so the
mapping survives Zig toolchain upgrades." A random matrix that can be
recomputed from a counter does not need to live in the checkpoint — the
same idea will return as dropout masks (Chapter 7), data-loader shuffles,
and the entire noise model of evolution strategies
([Chapter 9](09-training-without-gradients.md)). The (seed → values)
mapping is a *checkpoint contract*, which is why it must not depend on a
standard-library RNG whose sequence could change between Zig versions.

Also deliberately preserved: APOLLO's AdamW *fallback* reproduces the
reference's legacy-HuggingFace AdamW ("eps OUTSIDE the bias correction,
decay AFTER the step) — deliberately different from `AdamW` above",
`src/optim.zig:18-20`). Faithful porting means porting the reference's
inconsistencies too, because the goldens compare against the reference,
not against your aesthetic preferences.

### What they cost

`zig build bench-optim -Doptimize=ReleaseFast` measures step kernels at
Qwen3-0.6B-class shapes. The repo's snapshot (M1 Max, 2026-06-10, native
backend + Accelerate; docs/TRAINING.md §11 — a dated, machine-specific
measurement, *not* a general claim) for one 15.7M-param transformer block:
sgd 1.8–3.0 ms/step, adamw 5.2–6.0 ms, apollo-mini 23–24 ms, apollo-r256
44–46 ms, muon 446–457 ms. TRAINING.md's reading: the elementwise
optimizers run near memory bandwidth ("A full Qwen3-0.6B AdamW step costs
~200 ms of optimizer time — small next to the forward/backward"), Muon
pays GEMM FLOPs, APOLLO sits between. When someone tells you optimizer
choice is free, this table is the counterexample.

## 8.7 Param groups, clipping, and schedules

### Param groups are just optimizer instances

Real training recipes never use one hyperparameter set for everything —
the standard move is weight decay on matrices, none on biases and norms.
PyTorch models this with "param groups" inside one optimizer;
Fucina's observation (docs/TRAINING.md §4) is that a param group *is*
{hyperparams, params, state} — which is exactly what an optimizer instance
already is. So there is no group abstraction to learn: make two instances,
and let `OptimizerSet` make them feel like one (from TRAINING.md §4):

```zig
var decay = optim.AdamW.init(allocator, .{ .lr = 1e-3, .weight_decay = 0.1 });
var no_decay = optim.AdamW.init(allocator, .{ .lr = 1e-3, .weight_decay = 0 });
// ... addParam matrices to `decay`, biases/norms to `no_decay` ...
var set = optim.OptimizerSet.init(allocator);
try set.add(&decay);
try set.add(&no_decay);
// set.step / set.zeroGrad / set.clipGradNorm / set.saveState / set.loadState
```

Mixing optimizer *types* in one set works — Muon for the trunk, AdamW for
an adapter — because the set is type-erased through `AnyOptimizer`.

> **Zig note** — `AnyOptimizer` (`src/optim.zig:2426`) is dynamic dispatch
> without inheritance: a hand-rolled vtable. It stores a type-punned
> `ptr: *anyopaque` plus a `vtable: *const VTable` of function pointers
> (`step`, `zeroGrad`, `gradSquaredNorm`, `scaleGradients`, `saveState`,
> `loadState`); `anyOptimizer(opt: anytype)` builds the vtable at comptime
> for whatever concrete optimizer you hand it. This is the same pattern
> `std.mem.Allocator` uses, and it is the idiomatic Zig answer to "I need
> an interface": explicit, inspectable, and only where dynamic dispatch is
> actually required — everywhere else in this chapter, `anytype` +
> comptime monomorphization does the job with zero indirection.

`OptimizerSet.add` also fixes the double-step hazard from §8.5: it calls
the member's `collectGradStates` to check every parameter against all
previously-added members, so registering one variable into two groups
returns `error.DuplicateParam` instead of silently stepping it twice
(`docs/REFERENCE.md` §11.3).

### Clipping

`clipGradNorm(ctx, max_norm)` is `torch.nn.utils.clip_grad_norm_`
semantics: compute the global L2 norm over *all* registered params
(deterministic fixed-chunk f64 reduction); if it exceeds `max_norm`, scale
every gradient by `max_norm / (total + 1e-6)`; return the pre-clip norm —
which is worth logging, because a rising gradient norm is the classic
early warning of a diverging run. On an `OptimizerSet` the norm is global
across all groups, matching `clip_grad_norm_(model.parameters())`. And
note the exception that proves the recipes are read before being ported:
"the APOLLO recipes train with clipping disabled — its norm-growth limiter
replaces it" (docs/TRAINING.md §4).

### Schedules

A learning-rate schedule in this library is not an object hierarchy; it is
a *pure function of the step* plus a tiny hook that rescales `config.lr`
from a captured base:

```zig
try sched.attach(&opt.config.lr);   // captures the current lr as the base
sched.apply(optim.warmupCosineFactor(step_i, total_steps, warmup, 0.1));
```

The one built-in factor function, whole, from `src/optim.zig:2406-2415`:

```zig
pub fn warmupCosineFactor(step: u64, total_steps: u64, warmup_steps: u64, min_factor: f64) f64 {
    if (warmup_steps > 0 and step < warmup_steps) {
        return @as(f64, @floatFromInt(step + 1)) / @as(f64, @floatFromInt(warmup_steps));
    }
    if (total_steps <= warmup_steps) return min_factor;
    const num = @as(f64, @floatFromInt(step - warmup_steps));
    const den = @as(f64, @floatFromInt(total_steps - warmup_steps));
    const progress = @min(num / den, 1.0);
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + @cos(std.math.pi * progress));
}
```

Linear ramp from `1/warmup_steps` to 1, then a half-cosine down to
`min_factor`. Ten lines cover the entire "LR scheduler" concept, and the
purity is not a style point — it is what makes *resume* trivial: "Because
the factor is a pure function of the step, resuming from a checkpoint just
re-applies it — this is why lr is deliberately NOT a validated field of
optimizer checkpoints" (`docs/REFERENCE.md` §11.4). The checkpoint
validates structural config (a changed APOLLO rank is
`error.CheckpointConfigMismatch`) but not lr, because schedules
legitimately change lr every step.

> **ML note** — Why warmup at all? Early in training, Adam's `v` estimate
> is built from a handful of noisy gradients, and the parameters are at a
> random init where gradients are large and unrepresentative; full-size
> steps there can lock the run into a bad region. Ramping the lr over the
> first few hundred steps lets the moment estimates settle first. The
> cosine tail is gentler than step decays and has one fewer hyperparameter
> — which is most of why it became the default in LLM recipes.

## 8.8 Gradient accumulation: a fresh graph per backward

Suppose the batch you want does not fit in memory. The fix costs nothing,
because the substrate already does it (docs/TRAINING.md §4): `backward()`
**adds** into each parameter's persisted gradient, leaf gradients live
*outside* exec scopes, `step()` reads them non-destructively, and
`zeroGrad()` is the only clear. So a batch of N micro-batches is: N
forward+backward passes, then ONE clip, ONE step, ONE zeroGrad. Skipping
`zeroGrad` isn't a bug — it *is* gradient accumulation.

One rule has no workaround: **each backward needs a fresh graph**. A second
`backward()` over the *same* retained graph fails with
`error.BackwardAlreadyRun` (Chapter 7); accumulation is one backward per
freshly-built forward, which the recommended shape — one exec scope per
micro-batch — gives you naturally. Here is the real accumulation window
from `examples/finetune/main.zig:340-374` (trimmed; comments original):

```zig
// Exact token-weighted normalization: the samples differ in
// supervised-token counts, so mean-of-means (`.mean` + 1/N)
// would mis-weight them. `.sum` CE scaled by 1/total_valid makes
// the accumulated gradient — and the reported sum of scaled
// losses — the true mean over the window's supervised tokens.
...
const loss_scale = 1.0 / @as(f32, @floatFromInt(total_valid));
for (window) |idx| {
    // One exec scope per micro-batch: each graph is freed right
    // after its backward; the leaf grads accumulate outside the
    // scopes (backward ADDS until zeroGrad).
    const sample = &samples[idx];
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.lossExt(&ctx, sample.inputs, sample.labels, .{
        .reduction = .sum,
        .loss_scale = loss_scale,
    });
    try loss.backward(&ctx);
    loss_value += try loss.item();
    step_tokens += sample.inputs.len;
}
// ONCE per window: clip reads the full accumulated gradients
// (a mid-window clip would rescale partial sums).
_ = try set.clipGradNorm(&ctx, 1.0);
try set.step(&ctx);
set.zeroGrad();
```

Four disciplines hide in those lines, each documented in TRAINING.md §4:

- **Mean-of-means is not the batch mean.** `.mean` per micro-batch +
  averaging works only when every micro-batch has the same number of
  supervised tokens. With `ignore_index` masking, counts differ — so use
  `.sum` CE scaled by `1/total_valid`. The sum of the scaled losses then
  *is* the true mean over the window.
- **Normalization is loss-side and canonical.** Scaling the gradients
  afterward (`scaleGradients(ctx, 1.0/N)`) is deterministic too, but a
  different floating-point order: "NOT bitwise-equal to loss-side scaling.
  Pick one; tests here pin loss-side."
- **Clip ONCE, after the window** — a mid-window clip rescales a partial
  sum, "silently wrong", and there is a regression test for it
  (`src/optim_tests.zig`).
- **Checkpoint only at window boundaries.** Accumulated gradients live in
  the GradStates and are never serialized; a save mid-window silently
  drops the partial sum on resume. Watch for the two counters: the
  trainers' `step` counts *micro*-batches, the optimizer slots count
  *macro* steps; a resumable state has `step % accum_steps == 0`. LR
  schedules are keyed by the macro step.

## 8.9 Checkpoints and bit-exact resume

Training runs crash, machines reboot, experiments fork. Checkpointing in
this repo is built around one uncompromising definition of "it works": a
resumed run must produce **bit-for-bit** the same parameters as the run
that never stopped. Not "close", not "statistically equivalent" —
identical, testable with `==`. This section builds up the pieces; §8.11
shows the test that enforces it.

**Parameter values, two formats** (docs/TRAINING.md §8):

- `optim.saveTensors(writer, .{ &w1, &b1, ... })` / `loadTensors` —
  legacy positional **FZT1**, f32-only: the loader must list the same
  tensors in the same order. Trivially simple, brittle by construction.
- `optim.saveStateDict` / `loadStateDict` — **named** state dicts: per
  entry a unique name, dtype (f32/f16/bf16), shape, and bytes; load
  matches by name so entry order is free; strict mode (default) demands a
  one-to-one match. The file format underneath is plain **safetensors**,
  so a Fucina state dict opens in any safetensors tool
  ([Chapter 11](11-model-files-and-quantization.md)).

**Naming without boilerplate: `ParamRegistry`** (`src/param_registry.zig`).
Hand-listing every tensor twice (once for the optimizer, once for
persistence) is the kind of duplication that rots. `registry.collect(&model)`
reflects over your model struct at comptime, finds the tensor fields, and
names them by field path — the flat spirals `Model` yields `"w1"`, `"b1"`,
…; a nested model gets dotted paths automatically. One registry then feeds
both worlds: `saveStateDict`/`loadStateDict` for persistence and
`addParamsTo(opt)` for optimizer registration.

**Optimizer state.** `opt.saveState(writer)` / `loadState(reader)` persist
moments, step counts, and structural config fields in native frames
(magics like FZAD/FZM3/FZP3; FZO3 wraps an OptimizerSet). Slots are matched
by name — explicit via `addParamNamed`, otherwise the auto-name `param<i>`
from the slot's index — so unnamed params must keep their relative
registration order to reproduce their auto-names (docs/TRAINING.md §8).
Loads validate:
a changed APOLLO rank is `error.CheckpointConfigMismatch`; a bf16-state
checkpoint loaded into an f32-configured optimizer (or vice versa) is
`error.CheckpointDtypeMismatch`, because "There is deliberately NO implicit
f32<->bf16 conversion on load: it would silently break the bit-exact-resume
contract" (docs/TRAINING.md §8). A failed load leaves the target partially
restored — treat load errors as fatal for that instance.

**The directory protocol** (`src/training_checkpoint.zig`):

```text
checkpoint/
  model.safetensors        # or adapters.safetensors
  optimizer.fucina         # native optimizer frames
  trainer_state.json       # written last; commit sentinel
```

The JSON sentinel is written **last**, so "a directory with a parseable
`trainer_state.json` is a complete, committed checkpoint; a crash mid-save
leaves a sentinel-less directory" (docs/REFERENCE.md §11) that a loader
can recognize as uncommitted. Individual files are written atomically
(`writeFileAtomic`). This is a database commit protocol in three files —
no fsync heroics, just careful ordering.

**Why bit-exact resume is even possible** is the sum of choices you have
already met: element-independent parallel update maps (§8.5), pinned-order
norm reductions, serial row-order loss sums (§8.3), regenerated-not-stored
randomness (§8.6), and pure-function LR schedules (§8.7). The one
documented caveat (docs/TRAINING.md §8): a tensor receiving 3+ gradient
contributions whose consumers' heavy backward branches run async is
accumulated in completion order; two contributions are always order-safe.

> **ML note** — Why obsess over bitwise equality when training is
> stochastic anyway? Because determinism is the difference between
> debugging and guessing. If resume is bit-exact, then any divergence
> between two runs is a *real* change you made — not scheduler noise, not
> a lost partial gradient window. It also makes "did my refactor change
> the math?" a yes/no question answerable in one run. Determinism here is
> not a purity aesthetic; it is the repo's primary debugging instrument.

## 8.10 Mixed precision: 16-bit leaves, f32 gradients, f32 masters

Half-precision parameters halve your memory, and on some ISAs speed up the
GEMMs. They also introduce a genuinely subtle failure mode, and this
library's design walls it off with one contract (docs/TRAINING.md §10):

> **The contract: gradients are always f32.** Parameters and activations
> may be f16/bf16; every gradient in the engine is an f32 tensor
> (`GradState` holds f32 only). This is torch AMP's numeric contract, and
> it makes loss scaling unnecessary: loss scaling exists to rescue f16
> GRADIENT underflow, and gradients here never narrow.

If you have used PyTorch AMP you have met `GradScaler` — the machinery
that multiplies the loss by 2¹⁶ so f16 gradients do not flush to zero,
then divides it back out and skips steps when it overflows. All of that
exists to work around gradients being stored in f16. Keep gradients f32
and the entire apparatus evaporates.

The second half of the contract handles the *update*: an lr-sized nudge is
often smaller than a bf16 parameter can represent, so stepping 16-bit
storage directly would round most updates away. Fucina's answer is the
standard one, implemented in `Param` (`src/optim.zig:463-499`): 16-bit
params step through an **f32 master copy** the optimizer owns —

```zig
/// The f32 buffer the update kernels step: the param storage itself for
/// f32 params, the master for 16-bit params.
fn data(self: *Param) []f32 {
    return switch (self.value) {
        .f32 => |*t| t.data(),
        else => self.master,
    };
}
// … ensureMaster / refreshMasterFromValue elided …
/// Narrow the stepped master back into the 16-bit param storage (no-op
/// for f32 params).
fn publish(self: *Param) void {
    switch (self.value) {
        .f32 => {},
        .f16 => |*t| exec_convert.castF32ToF16(t.data(), self.master),
        .bf16 => |*t| exec_convert.castF32ToBf16(t.data(), self.master),
    }
}
```

— so sub-resolution updates *accumulate* in the master instead of rounding
away, and each step narrows the master back into the served storage. The
masters persist in v5 optimizer frames, keeping resume bit-exact
(docs/TRAINING.md §10). The recommended pattern is *16-bit leaves*: create
f16/bf16 variables, register them with any optimizer, and enter the graph
through the differentiable `to(.f32)` widen or through `dot`/`einsum` with
the 16-bit tensor as RHS. `examples/nanochat` trains its transformer
matrices this way (bf16 leaves, f32 embeddings; docs/TRAINING.md §10).

A separate, orthogonal knob — easy to confuse with the above — is 16-bit
**optimizer state**: storing the m/momentum buffers in bf16
(`state_dtype = .bf16`) while step math stays f32. Since the step is
memory-bound, narrower state is *faster*, not slower — the repo's dated
M1 Max snapshot (2026-07-03, docs/TRAINING.md §11) has the block AdamW
step at 3.605 → 2.028 ms with bf16 m. But AdamW's second moment gets its
own opt-in (`second_moment_dtype`) for a reason worth reading twice,
because it is a rare quantitative precision argument a newcomer can fully
follow (docs/TRAINING.md §3): with β₂ = 0.999 the second moment changes
~0.1% per step, which "sits BELOW bf16's ~2^-8 ≈ 0.39% resolution, so the
EMA can round to a no-op and stall". Scale matters too: full-param AdamW
on Qwen3-0.6B drops from 4.76 GB of state to 3.57 GB (m bf16) or 2.38 GB
(m+v bf16), while the default LoRA fine-tune "saves only ~4.6 MB of its
9.2 MB state — the knob matters at full-parameter/embedding scale, not for
small adapters" (docs/TRAINING.md §3).

> **Zig note** — Where does a dtype policy like "state may be bf16 but
> math stays f32" live in the code? In the type system, monomorphized: the
> `StateBuf = union(StateDType)` tagged union carries the buffer, the
> nested `switch` in §8.5 dispatches once per tensor, and
> `stateLoad`/`stateStore` widen/narrow at the boundary. And because "LLVM
> does NOT auto-vectorize these fused sqrt/div update loops on aarch64"
> (docs/TRAINING.md §11), the bf16 arms are hand-vectorized with
> `@Vector` (`state_vec_len`, `src/optim.zig`) — the portable-SIMD story
> from [Chapter 6](06-going-fast-on-cpus.md) paying off in the optimizer.
> Also note the comptime environment check guarding all of this
> persistence: `src/optim.zig:387-391` refuses to compile optimizer
> checkpoints on a big-endian target at all.

## 8.11 The payoff: spirals, end to end

Time to collect. `examples/spirals/main.zig` (493 lines) is the whole chapter
in one runnable file: a typed model struct, a forward pass, cross-entropy,
five optimizers, param groups with a schedule and clipping, checkpointing
halfway, and a resume that must be — literally, or the program errors —
bit-exact. The task is the classic two-spirals problem (Lang & Witbrock):
two interleaved spiral arms in the plane, one class each. It is tiny,
nonlinearly inseparable, and impossible to solve by accident — the course's
fruit fly, as promised in [Chapter 2](02-just-enough-ml.md).

### The model

```zig
const Model = struct {
    w1: Tensor(.{ .h1, .in }),
    b1: Tensor(.{.h1}),
    w2: Tensor(.{ .h2, .h1 }),
    b2: Tensor(.{.h2}),
    w3: Tensor(.{ .class, .h2 }),
    b3: Tensor(.{.class}),
```

*(from `examples/spirals/main.zig:29-35`)* — a model is a plain struct of typed
tensors. No `nn.Module`, no parameter registration ceremony; the tags from
[Chapter 4](04-axes-with-names.md) document the architecture in the types:
`w1` maps `.in` (2 coordinates) to `.h1` (64 hidden units), `w2` maps
`.h1` to `.h2`, `w3` maps `.h2` to `.class` (2 classes). Misconnect the
layers and it will not compile.

`Model.initRandom` (`examples/spirals/main.zig:38-67`) fills the weights with
scaled uniform noise, and builds the six tensors with an `errdefer` after
each — if the fourth allocation fails, the first three are freed, a
pattern you will now recognize everywhere in the repo. `initConstZero`
(`:70-90`) builds the *same struct* out of gradient-free constants — the
inference target for a finished checkpoint: same forward code, no autograd
overhead.

### Registration, forward, step

```zig
fn registerParams(opt: anytype, model: *Model) !void {
    try opt.addParam(&model.w1);
    try opt.addParam(&model.w2);
    if (comptime @hasDecl(@TypeOf(opt.*), "addFallbackParam")) {
        try opt.addFallbackParam(&model.w3);
    } else {
        try opt.addParam(&model.w3);
    }
    try opt.addParam(&model.b1);
    try opt.addParam(&model.b2);
    try opt.addParam(&model.b3);
}
```

*(from `examples/spirals/main.zig:116-127`)* — one function registers the model
with *any* optimizer, using comptime duck typing: if the optimizer type
declares `addFallbackParam` (Muon, APOLLO), the classifier head `w3` routes
to the fallback (§8.6 — heads must not be orthogonalized); otherwise it is
an ordinary param. Biases auto-route by rank. `@hasDecl` is resolved at
compile time, so the branch not taken never exists in the binary.

The forward and the step are the payoff for §8.2 — read the comment first,
it is the whole lifetime story (from `examples/spirals/main.zig:129-153`):

```zig
/// Forward pass inside an exec scope (ExecContext.openExecScope): every op result
/// is owned by the scope, so the eager autograd graph — whose nodes the
/// backward pass walks through raw pointers — stays alive until the scope
/// closes. No keeps, no defers: training forward code looks like inference.
fn forwardLogits(ctx: *ExecContext, model: *const Model, x: *const Tensor(.{ .batch, .in })) !Tensor(.{ .batch, .class }) {
    const z1 = try x.dot(ctx, &model.w1, .in);
    const s1 = try z1.add(ctx, &model.b1);
    const a1 = try s1.tanh(ctx);
    const z2 = try a1.dot(ctx, &model.w2, .h1);
    const s2 = try z2.add(ctx, &model.b2);
    const a2 = try s2.tanh(ctx);
    const z3 = try a2.dot(ctx, &model.w3, .h2);
    return try z3.add(ctx, &model.b3);
}

fn trainStep(ctx: *ExecContext, model: *const Model, x: *const Tensor(.{ .batch, .in }), labels: []const usize, opt: anytype) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope); // releases the whole step's graph
    const logits = try forwardLogits(ctx, model, x);
    const loss = try logits.crossEntropy(ctx, .class, labels);
    try loss.backward(ctx);
    try opt.step(ctx);
    opt.zeroGrad();
    return loss.item();
}
```

Eight op calls of pure inference-style code; one scope wrapping them; the
six-stage ritual from §8.1 (this tiny full-batch demo skips clipping — the
groups demo below adds it). `accuracy` (`:155-166`) reuses the *same*
`forwardLogits` under its own scope, takes `argmax` over `.class`, and
counts matches — the same-forward-infers-and-trains property, live.

### Data and checkpointing

`makeSpirals` (`examples/spirals/main.zig:168-186`) generates the dataset: 200
points per arm, radius growing with angle over ~1.75 turns, class 1 being
class 0 rotated by π, plus a whisper of Gaussian noise. Four hundred
points, two floats each — the whole dataset is a stack array.

`saveCheckpoint` / `loadCheckpoint` (`:188-228`) implement §8.9's directory
protocol with the real API: `beginSave`, `writeFileAtomic` for
`model.safetensors` (via `ParamRegistry.collect` — the reflective walk that
names tensors by field name, `:230-246`) and `optimizer.fucina` (via
`opt.saveState`), then `saveTrainerState` writing the JSON sentinel last.
Note the `comptime @TypeOf(opt) != @TypeOf(null)` guard: the same function
serves phase 3, which loads weights with no optimizer at all.

### The three-phase gauntlet

The `demo` driver (`examples/spirals/main.zig:260-325`) runs each optimizer
through three phases:

1. **Train** 2000 full-batch steps, checkpointing model + optimizer at
   step 1000.
2. **Resume**: build a *fresh* model — deliberately initialized from a
   different seed, so nothing can leak — and a fresh optimizer, load the
   checkpoint, retrain steps 1000–2000, and compare every final parameter
   against phase 1:

   ```zig
   var max_diff: f32 = 0;
   for (reference, replayed) |a, b| max_diff = @max(max_diff, @abs(a - b));
   ...
   if (max_diff != 0) return error.ResumeNotBitExact;
   ```

   *(from `examples/spirals/main.zig:309-314`)* — not a tolerance check. `!= 0`.
   One flipped bit anywhere in a thousand replayed steps — a stored moment
   rounded differently, a thread-order-dependent reduction, an lr factor
   applied off by one step — and the example *fails its build*. This is
   §8.9's whole argument compressed into one line: determinism is a
   testable property, so test it.
3. **Inference**: load the final weights into the gradient-free
   `initConstZero` model and report its accuracy — proving the checkpoint
   round-trips into a model with no autograd attached.

`main` (`:327-380`) runs the gauntlet for SGD (nesterov momentum), AdamW,
Muon (with a retuned fallback lr — the comment explains that the reference
default 3e-4 is tuned for LLM heads, not a toy head), APOLLO, and
APOLLO-Mini. Note the roster: plain `Adam` is not among the demos, so its
resume path is *not* covered by this gate — its behavior is pinned by the
unit and golden tests in `src/optim_tests.zig` instead. Then `groupsDemo`
(`:388-493`) composes the full §8.7 recipe — matrices in a weight-decayed
AdamW, biases in a no-decay AdamW, one `OptimizerSet`, a warmup-cosine
schedule attached to both lrs, clip at 1.0 — and pushes *that* composition
through the same halfway-checkpoint bit-exact-resume gauntlet.

Also worth a look in `main`: the allocator.

```zig
var gpa = std.heap.DebugAllocator(.{}){};
defer if (gpa.deinit() == .leak) @panic("leak");
```

*(from `examples/spirals/main.zig:328-329`)* — the example *panics if it leaks a
single allocation*. TRAINING.md §11's advice: train in ReleaseFast,
validate in Debug — the DebugAllocator catches lifetime mistakes.

### Run it

```sh
zig build spirals -Doptimize=ReleaseFast
```

You will see one header, then three lines per optimizer (the groups demo
prints two), in this format
(these are the actual `print` format strings from
`examples/spirals/main.zig:291-321,347` — the numbers are for your machine to
fill in):

```text
two spirals: {d} points, MLP 2-{d}-{d}-2 (tanh), full-batch, {d} steps

[{s}] trained {d} steps: loss {d:.4}  accuracy {d:.1}%
[{s}] resume from step {d}: max |delta param| = {d} ({s})
[{s}] inference from checkpoint: accuracy {d:.1}%
```

Watch the loss column: every optimizer drives it down, at visibly
different speeds and to visibly different depths — the optimizer sections
of this chapter, rendered as data. Watch the resume line say `bit-exact`
six times. Then break something on purpose, in phase 2 only so the phases
genuinely diverge: register `b1` and `b2` in swapped order for the
*resumed* optimizer (both are `[64]`, so shape validation passes but the
moments land in the wrong slots — the unnamed-param auto-name contract
from §8.9), or restart `groupsDemo`'s resumed loop counter at 0 instead of
`ckpt_step` (the schedule is a pure function of the step — key it wrong
and the lr trajectory differs). Either way, watch the gate catch you.

## 8.12 What we skipped, and where it lives

- **Activation checkpointing** — trading compute for backward memory —
  belongs to autograd and was covered in [Chapter 7](07-autograd.md);
  `fucina.checkpoint` composes with everything here (the LLM trainer's
  `--checkpoint-layers` uses it, with bitwise-equal gradients).
- **Training without gradients at all** — evolution strategies, the other
  half of `fucina`'s training story — is the [next
  chapter](09-training-without-gradients.md).
- **LoRA fine-tuning of a real quantized LLM**, the SFT data pipeline, and
  the fine-tune → merge → quantize → serve loop are
  [Chapter 15](15-training-llms-on-cpu.md); you have already seen its
  accumulation window (§8.8).
- **How gradients are *verified*** — finite differences, PyTorch-golden
  replicas, and causal real-model probes — is described in
  docs/TRAINING.md §9 and belongs to the verification story of
  [Chapter 16](16-the-craft.md). One honesty note if you run
  `zig build finetune -- --verify-grads` yourself: TRAINING.md documents a
  known open issue where the Q4_K_S Taylor-probe leg reads R ≈ 0.59 and
  the tool exits non-zero at the published commit, while the
  finite-difference and torch-golden checks still pass — the discrepancy
  is confined to that quantization-staircase-sensitive probe. The repo
  documents its own open issues; the course keeps them documented too.

## What you now know

- A training step is a fixed six-stage ritual — forward → loss →
  `backward` → `clipGradNorm` → `step` → `zeroGrad` — and every piece of
  training machinery serves one of those stages.
- Training inverts inference's deinit-ASAP discipline because graph nodes
  are single-owner; exec scopes make the training lifetime rule implicit,
  so the same forward code infers and trains.
- Cross-entropy's knobs (`ignore_index`, `reduction`, `label_smoothing`)
  are load-bearing for real training, and the fused
  `linearCrossEntropyExt` shows why owning your ops pays at LLM scale.
- SGD is one line; momentum is an EMA of gradients; AdamW is twelve lines
  of decoupled decay + two EMAs + bias correction, with f64 scalar prep —
  and each optimizer in the repo is a golden-parity port of its reference,
  quirks included.
- Muon orthogonalizes matrix updates via Newton-Schulz (paying GEMM
  FLOPs); APOLLO compresses Adam's moments into a regenerated random
  projection (paying almost nothing); both route non-matrix params to an
  embedded AdamW fallback.
- Param groups are just optimizer instances under an `OptimizerSet`;
  clipping is global-L2 with pre-clip norm returned; LR schedules are pure
  functions of the step, which is exactly what makes them resume-safe.
- Gradient accumulation is the substrate's default behavior plus
  discipline: fresh graph per backward, loss-side normalization, clip
  once, checkpoint at window boundaries.
- Checkpoints are a directory with a last-written JSON sentinel; state
  dicts are named safetensors via `ParamRegistry`; and bit-exact resume is
  an enforced, `error.ResumeNotBitExact`-gated property, not a hope.
- Mixed precision has one contract — gradients are always f32 — plus f32
  masters for 16-bit leaves; bf16 *optimizer state* is a separate,
  memory-bound-therefore-faster knob with one precision trap (AdamW's
  second moment).

## Explore the source

- `examples/spirals/main.zig` — the chapter as a program; read it top to bottom
  now, it will all be familiar.
- `docs/TRAINING.md` — the manual this chapter quotes; §12 is a pitfalls
  checklist worth keeping open while you write your first loop.
- `src/optim.zig` — all five optimizers, schedules, clipping, persistence;
  start at the module doc comment, then `adamwUpdate` and
  `newtonSchulz5`.
- `src/optim_tests.zig` — the golden parity tests against
  PyTorch/muon.py/apollo_torch; how "faithful port" is enforced.
- `src/param_registry.zig` and `src/training_checkpoint.zig` — reflective
  parameter naming and the checkpoint directory protocol, ~370 and ~290
  lines respectively.
- `examples/finetune/main.zig` — the same machinery at LLM scale: accumulation
  windows, OptimizerSet, periodic checkpoint directories.

## Exercises

1. **Break the ritual.** In `examples/spirals/main.zig`, move `opt.zeroGrad()`
   above `opt.step(ctx)` in `trainStep` and rerun. Explain what the
   optimizer now sees. Then delete `zeroGrad` entirely and explain why the
   loss behaves the way it does (§8.8 has the vocabulary).
2. **Add plain `Adam` to the gauntlet.** `main` demos five optimizers but
   not `optim.Adam`. Add a `demo(optim.Adam, "adam", …)` call with a
   sensible config and confirm it passes all three phases — you will have
   extended the bit-exact-resume gate to cover an optimizer it currently
   does not.
3. **A linear-warmup-only schedule.** Write your own factor function
   `fn warmupOnlyFactor(step: u64, warmup_steps: u64) f64` (course-code
   style, pure), use it in `groupsDemo` instead of `warmupCosineFactor`,
   and verify resume stays bit-exact — then explain *why* it must (§8.7).
4. **Momentum from scratch, verified.** Extend this chapter's course-code
   SGD with PyTorch's dampening and nesterov variants (the formulas are in
   the `torch.optim.SGD` docs), then check your implementation against
   `optim.SGD` on a few hand-built gradients. You are writing your first
   golden parity test.
5. **(Hard) A new optimizer behind the shared surface.** Implement
   Lion (Chen et al., 2023: sign of an interpolated momentum, decoupled
   decay) as a struct with the shared optimizer surface from §8.5
   (`init`/`deinit`/`addParam`/`step`/`zeroGrad`/`clipGradNorm`), reusing
   `optim.Param.of`. Wire it into `registerParams` and the spirals
   gauntlet. For full marks: make `saveState`/`loadState` round-trip and
   pass the bit-exact resume gate.

---

[Previous: Autograd — the graph hidden in the values](07-autograd.md) ·
[Next: Training without gradients](09-training-without-gradients.md)
