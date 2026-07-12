# Chapter 02 — Just enough machine learning

*Part I — Foundations*

You know Zig — or at least, after [Chapter 1](01-just-enough-zig.md), enough of it. This
chapter is the mirror image: exactly the machine learning you need before we start
building. No framework, no Python, no prior exposure assumed. By the end you will have
trained a model with a pencil, verified your arithmetic with `zig test`, and read — line
by line — the real training loop that the rest of this course spends fourteen chapters
rebuilding from nothing.

Here is the entire field in three sentences, which the rest of the chapter unpacks:

1. A **model** is an ordinary function with adjustable numeric parameters — knobs.
2. A **loss** is a single number measuring how wrong the model's output is.
3. **Training** is turning each knob a little in whichever direction makes the loss
   smaller, over and over, until it stops shrinking.

Everything else — tensors, layers, softmax, backpropagation, optimizers, transformers —
is machinery for doing those three things correctly, at scale, and fast.

## 2.1 A model is a function with knobs

Strip away the mystique and a machine-learning model is this:

```
output = f(input, parameters)
```

An ordinary, deterministic function. What makes it "learnable" is that some of its
inputs — the **parameters**, also called **weights** for historical reasons — are not
supplied by the caller but stored alongside the function, and we reserve the right to
change them. Same shape as a config struct threaded through your code, except nobody
hand-tunes the values: an algorithm does, guided by data.

Two vocabulary words that recur constantly:

- **Training** — the phase where the parameters change. Input *and* the desired output
  (the **label**) are both known; the mismatch between them drives the updates.
- **Inference** — the phase where the parameters are frozen and the function is simply
  called. This is "running" the model.

The smallest possible model has one knob. Take this one, which we will carry through
the whole chapter:

```
predict(x) = w · x
```

One parameter `w`, one input `x`, output is their product. Suppose we have exactly one
observation: when the input is `x = 2`, the correct answer is `y = 6`. You can solve
this in your head — `w` should be `3` — which is precisely why it is the right first
example: we will make a dumb iterative algorithm find the answer we already know, and
watch *how* it finds it. That algorithm, unchanged except in scale, is the one that
trains models with billions of parameters.

We start the knob in the wrong place: `w = 1`.

> **ML note** — Why "weights"? Early neural-network papers modelled a neuron's output as
> a *weighted sum* of its inputs, and the name stuck to all trainable parameters. You
> will also meet **bias** for an additive parameter (the `b` in `w·x + b`) — unrelated
> to statistical or social bias; it just shifts the output.

## 2.2 Loss: one number that measures wrongness

To improve the model we first need to say, numerically, how bad it is. A **loss
function** takes the model's prediction and the correct answer and returns a single
non-negative number: zero means perfect, larger means worse.

For predicting a number, the standard choice is **squared error**:

```
loss = (predict(x) − y)²
```

Why square the miss instead of, say, taking its absolute value?

- Squaring kills the sign — missing by −4 is as bad as missing by +4.
- It punishes large misses disproportionately: a miss of 4 costs 16, a miss of 0.1
  costs 0.01. The model is pushed hardest where it is most wrong.
- It is smooth everywhere. In a moment we will differentiate it, and smoothness is what
  makes that useful. (The absolute value has a kink at zero — its derivative
  jumps — which is exactly the kind of thing that makes optimization twitchy.)

Plug our model in and the loss becomes a function *of the knob*:

```
L(w) = (w·2 − 6)²
```

All of the arithmetic in this chapter is **course math** — worked by hand for the text,
rounded to at most four decimal places, and verified mechanically by the test block in
§2.4. A few values of `L`:

| `w` | prediction | loss |
|-----|-----------|------|
| 1.0 | 2.0 | 16.0 |
| 2.0 | 4.0 | 4.0 |
| 3.0 | 6.0 | 0.0 |
| 4.0 | 8.0 | 4.0 |
| 5.0 | 10.0 | 16.0 |

Plotted against `w`, this is a parabola with its bottom at `w = 3`. That picture — the
**loss landscape**, loss as a function of the parameters — is the single most useful
mental image in machine learning. Training is nothing more than walking downhill on
that surface. With one knob the landscape is a curve; with a million knobs it is a
million-dimensional surface you cannot picture, but the walk is the same.

Notice what we have already accomplished: the vague goal "make the model good" has
become the precise goal "find the `w` that minimizes `L(w)`". Every training run you
will ever see — spirals, guitar amps, language models — is minimizing some scalar
`L(parameters)`.

## 2.3 Downhill: the derivative tells you which way to turn

Standing at `w = 1` on that parabola, which way is downhill? You could probe: evaluate
`L(1.001)`, compare with `L(1)`, and see whether the tiny nudge helped. That works, and
it even has a name we will meet again (finite differences, §2.4). But calculus hands us
the answer directly: the **derivative** `dL/dw` is the slope of the loss curve at the
current `w` — how fast loss changes per unit change of the knob.

For our loss, expand and differentiate (or use the chain rule, which we will do
properly in §2.8):

```
L(w)    = (2w − 6)²
dL/dw   = 2 · (2w − 6) · 2  =  8w − 24
```

At `w = 1`: `dL/dw = −16`. Read it like an instrument:

- The **sign** says which way the ground tilts. Negative slope means loss *decreases*
  as `w` increases — so downhill is to the right: increase `w`.
- The **magnitude** says how steep it is. 16 is steep; near the bottom the slope
  approaches 0.

That yields the entire update rule of deep learning, called **gradient descent**:

```
w ← w − lr · dL/dw
```

Subtracting the slope moves against the tilt — downhill in both cases (negative slope →
`w` grows; positive slope → `w` shrinks). The knob `lr` is the **learning rate**: how
big a step to take. It is the first — and most consequential — of the *hyperparameters*,
the settings *you* choose rather than the training algorithm (step counts, layer sizes,
and learning rates are all hyperparameters).

Let us run it, by hand, with `lr = 0.1`. One **step** = compute prediction, loss,
derivative, then update `w`:

| step | `w` | prediction | loss | `dL/dw` | new `w` |
|------|--------|-------|----------|--------|--------|
| 0 | 1.0 | 2.0 | 16.0 | −16.0 | 2.6 |
| 1 | 2.6 | 5.2 | 0.64 | −3.2 | 2.92 |
| 2 | 2.92 | 5.84 | 0.0256 | −0.64 | 2.984 |
| 3 | 2.984 | 5.968 | 0.001024 | −0.128 | 2.9968 |

Work through step 0 yourself once — prediction `1·2 = 2`, error `2 − 6 = −4`, loss
`(−4)² = 16`, slope `8·1 − 24 = −16`, update `1 − 0.1·(−16) = 2.6` — and the rest of
the table follows the same rhythm. Three things worth staring at:

- **It converges, and it decelerates.** Each step closes exactly 80% of the remaining
  distance to `w = 3` (the gap shrinks 2 → 0.4 → 0.08 → 0.016). Steps are big where the
  landscape is steep and shrink automatically near the bottom, because the slope itself
  shrinks. Nobody schedules that; it falls out of the rule.
- **The loss collapses fast.** 16 → 0.64 → 0.0256: each step cuts loss by 25× here
  (the distance factor 0.2, squared). Real losses are not parabolas and fall less
  cleanly, but "fast early, slow late" is the shape of nearly every training curve you
  will ever plot.
- **We never solved for `w`.** We only ever evaluated the function and its slope
  locally, then nudged. That locality is why the same procedure scales to functions
  with 10⁹ knobs and no closed-form anything.

Now break it. The learning rate trades speed against stability, and this tiny model is
transparent enough to show the whole spectrum (course math again — exercise 2 asks you
to verify it):

| `lr` | behaviour |
|------|-----------|
| 0.1 | converges; gap ×0.2 per step (the table above) |
| 0.125 | lands on `w = 3` in **one step** — a fluke of parabolas, nothing more |
| 0.25 | oscillates forever: 1 → 5 → 1 → 5, loss stuck at 16 |
| 0.3 | **diverges**: 1 → 5.8 → −0.92 → …, loss 16 → 31.36 → 61.47, growing every step |

Too small wastes steps; too large overshoots the valley and, past a threshold, each
overshoot lands on ground *steeper* than the last, so the next leap is bigger —
runaway. When a real training run's loss suddenly reads `NaN`, an oversized step (or
its numerical cousins) is the first suspect. Half the craft of training is choosing and
scheduling `lr`; [Chapter 8](08-training.md) covers warmup and cosine decay, which is
"start gentle, cruise, then ease off" encoded as a pure function of the step number.

> **ML note** — Real loss landscapes are not bowls: they have valleys, plateaus, saddle
> points, and countless local dips. It remains one of the field's happiest empirical
> facts that plain gradient descent (plus the refinements of
> [Chapter 8](08-training.md): momentum, Adam) finds *good* parameters anyway. Nobody
> fully knows why in general — and honest practitioners say so.

## 2.4 The whole algorithm, compile-checked

Everything so far, as runnable Zig. This is **course code** — it lives in this course,
not in the Fucina tree — and it compiles and passes with the pinned toolchain
(`zig test one_knob.zig`, Zig 0.16.0). No library, no allocator, nothing but arithmetic:

```zig
const std = @import("std");

test "one knob, learned by gradient descent" {
    const x: f32 = 2.0; // the input we observe
    const y: f32 = 6.0; // the answer we want
    var w: f32 = 1.0; // the knob, starting in the wrong place

    var loss: f32 = 0;
    for (0..20) |_| {
        const pred = w * x; // forward pass: run the model
        const err = pred - y; // signed miss
        loss = err * err; // loss: squared error
        const grad = 2 * err * x; // dL/dw, by the chain rule
        w -= 0.1 * grad; // the descent step (lr = 0.1)
    }
    try std.testing.expect(loss < 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), w, 1e-3);
}

test "too large a learning rate diverges" {
    const x: f32 = 2.0;
    const y: f32 = 6.0;
    var w: f32 = 1.0;
    for (0..10) |_| {
        const grad = 2 * (w * x - y) * x;
        w -= 0.3 * grad; // each step now overshoots the minimum
    }
    // After ten steps we are farther from w = 3 than where we started.
    try std.testing.expect(@abs(w - 3.0) > 2.0);
}
```

The five commented lines of the first loop body are, in embryo, every training step in
this book. The names to internalize:

- Running the model on an input is the **forward pass** (`pred = w * x`).
- Computing `dL/dw` is — once models get deep — the **backward pass**
  ([Chapter 7](07-autograd.md) builds the machinery that does it automatically).
- The update line is the **optimizer** ([Chapter 8](08-training.md) builds five of
  them; this one-liner is SGD, *stochastic gradient descent*, the ancestor of them all).

One more test, because it plants a habit this repository takes unusually seriously:
*how do you know your derivative is right?* You check it against a slope measured
numerically — nudge the input both ways, divide rise by run. This is the **finite
differences** method:

```zig
test "finite differences agree with the hand-derived gradient" {
    const lossAt = struct {
        fn f(w: f64) f64 {
            const err = w * 2.0 - 6.0;
            return err * err;
        }
    }.f;
    const w: f64 = 1.0;
    const h: f64 = 1e-4;
    const numeric = (lossAt(w + h) - lossAt(w - h)) / (2 * h);
    const analytic = 2 * (w * 2.0 - 6.0) * 2.0;
    try std.testing.expectApproxEqAbs(analytic, numeric, 1e-6);
}
```

Slow (one extra forward pass per parameter per probe) but nearly impossible to get
wrong — which makes it the perfect *oracle* for the fast method. Fucina uses exactly
this idea, industrial-strength, to validate its automatic gradients: central
differences through an entire LLM fine-tuning loss agree with the analytic backward
pass at "cosine similarity 0.999998, worst |dev| 1.9e-4" (docs/TRAINING.md §9,
"Gradient verification — how we know the gradients are right"). Gradients are
*verified*, never trusted — a theme
[Chapter 7](07-autograd.md) returns to.

> **Zig note** — If the test syntax is unfamiliar: `test "name" { ... }` blocks are
> compiled and run by `zig test file.zig`, `try` propagates the error union that
> `std.testing.expect` returns, and the `struct { fn f(...) ... }.f` idiom is how you
> write a small local function inside a test. All covered in
> [Chapter 1](01-just-enough-zig.md).

## 2.5 Tensors: the one data structure

Our model had one parameter and one input. Real models have millions of parameters and
consume inputs like "400 points in the plane" or "three seconds of audio" or "8,192
tokens of text". Machine learning packs *all* of it — inputs, outputs, parameters,
intermediate results — into a single data structure: the **tensor**.

A tensor is an n-dimensional array of numbers plus its **shape**:

| rank | shape (example) | what it might hold |
|------|--------|--------------------|
| 0 | `{}` | a loss value |
| 1 | `{64}` | one layer's biases |
| 2 | `{400, 2}` | 400 points, 2 coordinates each |
| 2 | `{64, 2}` | a weight matrix: 64 outputs × 2 inputs |
| 3 | `{8, 512, 64}` | 8 attention heads × 512 positions × 64 features |

"Tensor" sounds exotic; "multi-dimensional array with a shape attached" is the whole
truth of it as a data structure. (The physicist's tensor carries extra geometric
meaning that ML cheerfully ignores.) What earns tensors their central place is not the
container but the *style of computation* it enforces: you do not loop over points and
process them one at a time — you apply one operation to the whole block at once.
"Multiply this `{400, 2}` by that `{64, 2}ᵀ`." That style has two payoffs:

- **Speed.** One bulk operation over contiguous memory is exactly what SIMD units and
  multiple cores want to eat. [Chapter 6](06-going-fast-on-cpus.md) is entirely about
  cashing this cheque.
- **Batching.** Feeding the model 400 inputs stacked into one tensor — a **batch** —
  amortizes every per-operation cost across the batch, and the resulting gradient is
  averaged over all 400 examples, which makes the downhill signal steadier than any
  single example's. When the batch is the *whole* dataset, training is **full-batch** —
  the spirals example runs this way; feeding it in random slices instead is
  *mini-batch*, and is where the "stochastic" in SGD comes from.

Here is the idea in the wild — the spirals example packing 400 (x, y) points into one
rank-2 tensor (from `examples/spirals.zig:344`):

```zig
var x = try Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ n_points, 2 }, &xs);
```

Note `.{ .batch, .in }`: in Fucina the axes of a tensor's shape carry *names*, checked
at compile time, and `Tensor(.{ .batch, .in })` is a different *type* from
`Tensor(.{ .in, .batch })`. That design — why shape bugs are the chronic pain of tensor
code and how naming axes removes a whole class of them — gets its own chapter
([Chapter 4](04-axes-with-names.md)); building the tensor itself, its storage,
refcounting, and views, is [Chapter 3](03-tensors-from-scratch.md). For now: tensor =
array + shape, and every value we compute with from here on is one.

> **ML note** — The numbers inside are almost always 32-bit floats (`f32`) during
> training. Squeezing them into 16, 8, 4, even ~2 bits per value — **quantization** — is
> what lets large models fit in ordinary RAM, and is the subject of
> [Chapter 11](11-model-files-and-quantization.md) and
> [Chapter 14](14-the-low-bit-frontier.md).

## 2.6 Layers: linear maps with a bend between them

Scale the one-knob model up. With `n` inputs and `m` outputs, the natural
generalization is: every output is a weighted sum of every input, plus an offset —

```
output = W·x + b        W: an {m, n} matrix of knobs, b: m more knobs
```

This is a **linear layer** (also *fully-connected* or *dense* layer), and it is the
workhorse of all deep learning. Each of its `m` output rows — one weight per input plus
a bias — is what older literature calls a *neuron*. The matrix multiply computing all
`m` weighted sums at once is where virtually all of a model's arithmetic lives, which
is why [Chapter 6](06-going-fast-on-cpus.md) spends most of its pages making one
operation — GEMM, general matrix multiply — fast.

But there is a trap: stack two linear layers and you have gained nothing.

```
f₁(x) = A·x + a
f₂(h) = B·h + b
f₂(f₁(x)) = B·(A·x + a) + b = (B·A)·x + (B·a + b)
```

`B·A` is just another matrix: the composition of two linear maps is a linear map. A
hundred stacked linear layers still can only draw straight lines (in classification
terms: only cut the input space with flat planes). All the depth collapses.

The fix costs one line: between the linear layers, insert a fixed, *nonlinear*
elementwise function — an **activation function**. Classic choices:

- `tanh(x)` — squashes to (−1, 1); smooth, symmetric; the choice in our spirals model.
- `relu(x) = max(0, x)` — zero for negatives, identity for positives; brutally simple
  and the modern default in most places.

With a bend between them, layers stop collapsing: the first layer's straight cuts get
bent by the activation, the next layer cuts the *bent* space, and a few rounds of
cut-and-bend can carve extraordinarily intricate regions. Enough hidden units can
approximate essentially any reasonable function — that is a theorem for one hidden
layer (with all the usual fine print), and in practice depth buys far more than width.

The stack "linear → activation → linear → activation → … → linear" is a **multi-layer
perceptron** (MLP), the oldest and simplest deep architecture. The layers between input
and output are **hidden layers** — hidden because nothing in the training data says
what their outputs should be; the training process invents useful intermediate
representations on its own.

Here is a real one — the forward pass of the spirals model, verbatim from
`examples/spirals.zig:129-142`:

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
```

Read it with your new vocabulary: `dot` is the matrix multiply (it contracts over the
axis you *name* — `.in`, `.h1`, `.h2` — instead of an axis you count; that is the
named-axes system of [Chapter 4](04-axes-with-names.md)); `add` applies the bias;
`tanh` is the bend. Three linear layers, two bends: input `{batch, 2}` → hidden
`{batch, 64}` → hidden `{batch, 64}` → output `{batch, 2}`. One knob has become
`64·2 + 64 + 64·64 + 64 + 2·64 + 2 = 4,482`, and *nothing about training changes*:
loss is still one number, and every knob still gets nudged downhill.

Don't worry yet about `ctx`, `try`, or the ominous doc comment about scopes and graphs
— those are precisely the machinery of Chapters [3](03-tensors-from-scratch.md),
[5](05-the-operation-library.md), and [7](07-autograd.md). What matters today is that
you can now read the *math* of this function.

## 2.7 Classification: softmax and cross-entropy

Our one-knob model predicted a *number* — that is **regression**. The spirals task asks
a different kind of question: given a point, *which of two spirals does it belong to?*
Predicting a category from a fixed set is **classification**, and it needs two new
ideas, both of which you can compute by hand.

**From scores to probabilities: softmax.** The network's final layer emits one raw
score per class — the function above is called `forwardLogits` because raw class scores
are called **logits**. Logits are unbounded and unnormalized: `{2.0, 0.5}` says "class
0 looks stronger", but how much stronger? The **softmax** function turns scores into a
probability distribution: exponentiate each score, then divide by the sum.

Course math, with logits `{2.0, 0.5}` (verified in the test below):

```
e^2.0 = 7.3891      e^0.5 = 1.6487      sum = 9.0378

p₀ = 7.3891 / 9.0378 = 0.8176
p₁ = 1.6487 / 9.0378 = 0.1824
```

The outputs are positive, sum to 1, and preserve order — the biggest logit gets the
biggest probability. Exponentiation makes the mapping aggressive: a logit *gap* of 1.5
became an odds ratio of about 4.5 to 1, and each extra unit of gap multiplies the odds
by `e ≈ 2.718` again. Two properties worth noting now, cashed in later:

- Adding a constant to *every* logit changes nothing (it cancels in the division).
  Production kernels exploit this by subtracting the max logit first so that `e^x`
  never overflows — you can see the three-pass max/exp-sum/normalize structure in
  Fucina's SIMD row kernel (`softmaxRows`, `src/exec/row_ops.zig:1249`). Numerical
  care of this kind is a recurring character in [Chapter 5](05-the-operation-library.md).
- Only gaps matter. Softmax is a smooth argmax — hence the name.

**From probabilities to a loss: cross-entropy.** Now the loss. For classification the
standard is **cross-entropy**: *the negative log of the probability the model assigned
to the correct class*.

```
loss = −ln(p_true)
```

With our numbers: if the true class is 0, `loss = −ln(0.8176) = 0.2014` — mild, the
model leaned the right way. If the true class is 1, `loss = −ln(0.1824) = 1.7014` —
much worse. The logarithm gives the loss its teeth: as `p_true → 1`, loss → 0; as
`p_true → 0`, loss → ∞. Being *confidently wrong* is punished without bound, so the
gradient pushes hardest exactly there. Over a batch, per-example losses are averaged.

Why not optimize accuracy directly — the thing we actually care about? Because accuracy
is a staircase: nudge a knob infinitesimally and the predicted class almost never
flips, so accuracy's derivative is zero almost everywhere. Zero slope, no downhill, no
signal. Cross-entropy is the smooth stand-in that *does* slope — it improves whenever
any correct-class probability rises, long before the argmax flips. Choosing losses that
are smooth surrogates for what you truly want is one of the field's central design
moves.

Two anchors to carry:

- **The know-nothing baseline is ln(K).** A model emitting equal logits for K classes
  assigns probability 1/K everywhere, so its cross-entropy is `−ln(1/K) = ln K`. For
  two classes that is `ln 2 ≈ 0.693` — coin-flip loss. A freshly initialized classifier
  should start near it; meaningfully above means something is broken. Fucina pins this
  fact as a machine-verified test: uniform logits over 4 classes yield exactly
  `ln 4` (docs/REFERENCE.md §4.15, test "crossEntropy on uniform logits is ln(K)").
- **In practice softmax + cross-entropy is one fused operation.** Numerically and for
  efficiency they are computed together; in Fucina the model's logits go straight into
  `crossEntropy(ctx, .class, labels)` — signature in docs/REFERENCE.md §4.15 — and no
  probability tensor is ever materialized unless you ask for one.

Here is the by-hand arithmetic above as course code (the fourth test in this chapter's
compile-checked file):

```zig
test "softmax and cross-entropy, by hand" {
    const logits = [2]f32{ 2.0, 0.5 };
    var exps: [2]f32 = undefined;
    var sum: f32 = 0;
    for (logits, &exps) |l, *e| {
        e.* = @exp(l);
        sum += e.*;
    }
    const p0 = exps[0] / sum;
    const p1 = exps[1] / sum;
    try std.testing.expectApproxEqAbs(@as(f32, 0.8176), p0, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1824), p1, 1e-4);
    // Cross-entropy = -ln(probability assigned to the true class).
    try std.testing.expectApproxEqAbs(@as(f32, 0.2014), -@log(p0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.7014), -@log(p1), 1e-4);
}
```

> **ML note** — The same recipe scales absurdly well. A language model is "just" a
> classifier over its vocabulary: given the text so far, softmax over ~150,000 possible
> next tokens, cross-entropy against the token that actually came next. The loss you
> will meet in [Chapter 12](12-a-transformer-from-scratch.md) is this section with a
> bigger K.

## 2.8 Many knobs at once: gradients and the chain rule

Two gaps remain between our one-knob toy and the 4,482-knob spirals model.

**Gap one: many knobs.** With parameters `w₁ … wₙ`, the derivative generalizes to the
**gradient**: the vector of **partial derivatives** `(∂L/∂w₁, …, ∂L/∂wₙ)`, each
answering "if I nudged *only this knob*, how would the loss change?" The update rule
survives untouched — subtract `lr` times the gradient, every knob simultaneously:

```
wᵢ ← wᵢ − lr · ∂L/∂wᵢ        for every i at once
```

The gradient vector points in the direction of steepest *ascent* of the loss surface;
stepping against it is the steepest way down. Everything you learned in §2.3 — signs,
step sizes, divergence — applies coordinate by coordinate.

**Gap two: computing a million partials without a million passes.** Finite differences
would need at least one extra forward pass *per knob per step* — at a million knobs,
a million forward passes to take one step. Dead end. The rescue is the **chain rule**,
the one piece of calculus this course leans on, so let us do it concretely.

Break our familiar loss into its actual computational steps:

```
p = w · x        (prediction)         dp/dw = x
e = p − y        (error)              de/dp = 1
L = e²           (loss)               dL/de = 2e
```

The chain rule says: to find how `w` affects `L` through the chain, *multiply the local
slopes along the path*:

```
dL/dw = dL/de · de/dp · dp/dw = 2e · 1 · x
```

At `w = 1`: `e = −4`, so `dL/dw = 2·(−4)·1·2 = −16`. The same −16 as §2.3 — but notice
*how* we got it: not by algebraically expanding `L(w)` (hopeless for a real network),
but by walking the chain of operations *backwards from the loss*, multiplying one local
derivative per step. Each local derivative is trivial — every primitive operation
(multiply, add, tanh, softmax…) knows its own slope. The structure does the rest.

Now the payoff. A neural network's forward pass is exactly such a chain — a graph of
primitive operations. Walk it once backwards, and partial derivatives for *all*
parameters fall out along the way, at a total cost of a small constant multiple of one
forward pass. Two knobs or two billion: one backward sweep. This algorithm is
**backpropagation**, and the machinery that runs it automatically — recording, for each
value, which operation produced it, then walking those records in reverse — is called
**autograd**. It is why line four of our §2.4 loop (`grad = 2 * err * x`, hand-derived)
is the *only* line that doesn't scale as written, and why in the real `trainStep` below
it is replaced by a single call: `loss.backward(ctx)`.

Building an autograd engine — first a ~100-line scalar one, then Fucina's, where the
live tensors themselves *are* the graph — is [Chapter 7](07-autograd.md). Until then
you may treat `backward()` as certified magic: certified, because §2.4's finite
differences are exactly how the certification works.

## 2.9 Two spirals: the course's fruit fly

Biologists study *Drosophila* not out of love for flies but because it is small, fast
to breed, and shows real genetics. This course needs the same thing: a task small
enough to train in seconds on a laptop CPU, transparent enough to plot on paper, and
just hard enough that solving it proves the machinery genuinely works. Ours is the
**two-spirals problem** — the repo's own comment calls it "the classic Lang &
Witbrock task" (`examples/spirals.zig:168`), a benchmark from the late-1980s
connectionist era that was famously obnoxious for the small networks of the day.

The task: points are scattered along two interleaved spiral arms in the plane. Given a
point's `(x, y)` coordinates, say which arm it belongs to. Here is the dataset
generator, verbatim from `examples/spirals.zig:168-186`:

```zig
/// Two interleaved spirals (the classic Lang & Witbrock task): radius grows
/// with the angle over ~1.75 turns; class 1 is class 0 rotated by pi.
fn makeSpirals(seed: u64, xs: *[n_points * 2]f32, labels: *[n_points]usize) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();
    for (0..n_per_class) |i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, n_per_class - 1);
        const theta = t * 3.5 * std.math.pi;
        const r = 0.15 + 0.85 * t;
        const x = r * @sin(theta);
        const y = r * @cos(theta);
        xs[4 * i + 0] = x + random.floatNorm(f32) * 0.02;
        xs[4 * i + 1] = y + random.floatNorm(f32) * 0.02;
        labels[2 * i] = 0;
        xs[4 * i + 2] = -x + random.floatNorm(f32) * 0.02;
        xs[4 * i + 3] = -y + random.floatNorm(f32) * 0.02;
        labels[2 * i + 1] = 1;
    }
}
```

The idea in three lines of math: sweep a parameter `t` from 0 to 1; place a point at
angle `θ = 3.5π·t` (1¾ turns) and radius `r = 0.15 + 0.85·t` (growing as it turns —
that is what makes a spiral); the second class is the same point negated, i.e. rotated
by 180°, so the two arms thread perfectly between each other. A pinch of Gaussian noise
(±0.02-ish) keeps it honest. 200 points per arm, 400 total — and note `seed`: same
seed, same dataset, every run, on every machine. Determinism as a habit starts at the
data.

Why this task earns fruit-fly status:

- **A linear model cannot do it, at all.** No straight line separates two interleaved
  spirals — each arm wraps around the other, so any line you draw has both classes on
  both sides, and a linear classifier can only draw lines (§2.6). The task is
  *structurally* beyond linear; hidden layers aren't an optimization here, they are the
  difference between possible and impossible. Exercise 5 makes you feel this.
- **It is two-dimensional.** You can scatter-plot the data, and even plot the trained
  model's decision regions, and *see* what 4,482 knobs learned (exercise 4).
- **It is small.** Full-batch training of the 2-64-64-2 tanh MLP finishes in seconds,
  so you can afford to re-run it dozens of times while experimenting — which is the
  whole point of a model organism.
- **It exercises everything.** Tensors, a real multi-layer forward pass, softmax +
  cross-entropy, backward, optimizer — nothing important is skipped, nothing
  distracting is added.

And here is the training step it all feeds, verbatim from `examples/spirals.zig:144-153`
— compare it to your five-line loop from §2.4:

```zig
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

Line by line, with what you now know: run the forward pass (§2.6); fuse softmax and
cross-entropy into one scalar loss (§2.7); backpropagate to fill every parameter's
gradient (§2.8); let the optimizer apply the update rule to all 4,482 knobs (§2.3);
zero the gradients so the next step starts clean. The scope lines are memory
management — Zig has no garbage collector, and who owns the step's intermediate tensors
is a real question with an elegant answer ([Chapter 7](07-autograd.md)).

Run it yourself:

```
zig build spirals -Doptimize=ReleaseFast
```

(the file's header comment recommends exactly this; `examples/spirals.zig:14`). The
demo trains the MLP with each of five optimizers in turn, printing the final loss and
accuracy for each — and, for each, it also does something that tells you more about
this library's character than any feature list: it saves a mid-training **checkpoint**
(a snapshot of all parameters plus optimizer state), restores it into a *fresh* model,
retrains the second half, and *demands the final parameters match the original run bit
for bit* —
`if (max_diff != 0) return error.ResumeNotBitExact;` (`examples/spirals.zig:314`).
Not "close". Identical, to the last bit of every float. That training is exactly
reproducible — same data, same seed, same thread configuration, across runs and
across checkpoint resumes — is a design contract here (`docs/TRAINING.md` §4's
determinism contract), and [Chapter 8](08-training.md) shows what it costs to keep. Watch the loss values fall
as it runs; we will account for every single line of this file by the end of
[Chapter 8](08-training.md).

The spirals return throughout the course: trained without gradients in
[Chapter 9](09-training-without-gradients.md) (`examples/es_spirals.zig`), and with
ternary {−1, 0, +1} weights in [Chapter 14](14-the-low-bit-frontier.md)
(`examples/ptqtp_spirals.zig`, `examples/es_ternary_spirals.zig`). Same fly, new
microscopes.

## 2.10 The road ahead

Every concept this chapter introduced by intuition gets built, from empty file to
tested code, in the chapters ahead:

| you met | we build it in |
|---------|----------------|
| tensors as shape + numbers | [Chapter 3 — Tensors from scratch](03-tensors-from-scratch.md) |
| axes with names, `Tensor(.{ .batch, .in })` | [Chapter 4 — Axes with names](04-axes-with-names.md) |
| `dot`, `add`, `tanh`, softmax, cross-entropy | [Chapter 5 — The operation library](05-the-operation-library.md) |
| why bulk tensor ops are fast | [Chapter 6 — Going fast on CPUs](06-going-fast-on-cpus.md) |
| `backward()` — the chain rule, automated | [Chapter 7 — Autograd](07-autograd.md) |
| `opt.step()`, learning rates, checkpoints | [Chapter 8 — Training](08-training.md) |
| training with *no* gradients at all | [Chapter 9 — Training without gradients](09-training-without-gradients.md) |

Then the applications: a real-time neural guitar amp
([Chapter 10](10-the-guitar-amp.md)), and language models — files, transformers,
inference tricks, low-bit weights, CPU fine-tuning (Chapters
[11](11-model-files-and-quantization.md)–[15](15-training-llms-on-cpu.md)). One
promise, kept throughout: nothing stays magic. Every operation called in
`forwardLogits`, every line of `trainStep`, will have been built in front of you, in
Zig, with tests proving it right.

## What you now know

- A model is a function with adjustable parameters; training adjusts them, inference
  runs them frozen.
- A loss function condenses "how wrong is the model" into one non-negative number;
  training is minimizing loss as a function of the parameters — walking downhill on the
  loss landscape.
- The derivative's sign points downhill and its magnitude sets the natural step;
  `w ← w − lr·dL/dw` is gradient descent, and you executed it by hand: 16 → 0.64 →
  0.0256 → 0.001 on the one-knob model.
- The learning rate trades speed against stability: too small crawls, too large
  oscillates or diverges (you watched `lr = 0.3` blow up the same problem).
- Tensors — arrays with shapes — hold everything, and batching turns per-example work
  into bulk operations that hardware loves.
- A linear layer is `W·x + b`; stacked linears collapse into one, so activation
  functions (tanh, relu) between them are what buy depth its power.
- Classification = logits → softmax (scores to probabilities) → cross-entropy
  (−ln p_true); `ln K` is the know-nothing baseline (≈ 0.693 for two classes), and
  accuracy can't be a loss because it has no slope.
- The gradient is the vector of partial derivatives; the chain rule computes all of
  them in one backward sweep (backpropagation), for any number of parameters — and
  gradients are verified against finite differences, never trusted.
- The two-spirals task is the course's model organism: impossible for linear models,
  plottable, trained in seconds — and `examples/spirals.zig` is the file this course
  spends Part II and III rebuilding piece by piece.

## Explore the source

- `examples/spirals.zig` — this entire chapter as one runnable program: dataset
  generator (line 168), model struct with named-axis tensors (line 29), forward pass
  (line 133), training step (line 144), and the bit-exact-resume check (line 314). Read
  it top to bottom; measure how much of it you can already follow.
- `docs/TRAINING.md` §1 — "A complete training step": the full six-stage ritual
  (forward → loss → backward → clip → step → zeroGrad); `trainStep` instantiates
  five of the six (the spirals `groupsDemo` adds the clip).
- `docs/REFERENCE.md` §4.15 — the loss-function catalogue, including the
  machine-verified "crossEntropy on uniform logits is ln(K)" test you can now derive
  yourself.
- `src/exec/row_ops.zig` (`softmaxRows`, line 1249) — softmax as production code:
  find the max-subtraction trick from §2.7 inside the SIMD loops. Skim only; this is
  Chapter 5–6 territory.

## Exercises

1. **(easy)** In the §2.4 descent test, change the data point to `x = 4, y = 6`.
   Before running: what is the optimal `w`, and what is `dL/dw` at `w = 1`? Does
   `lr = 0.1` still converge? (Work out the gap-shrink factor `|1 − lr·2x²|` and check
   your prediction against the test.)
2. **(easy)** Verify the learning-rate table of §2.3 by hand or by adapting the test:
   confirm that `lr = 0.125` converges in one step, that `lr = 0.25` cycles between
   `w = 1` and `w = 5` forever, and that `lr = 0.3` produces the losses
   16 → 31.36 → 61.4656. Why is one-step convergence special to quadratic losses?
3. **(medium)** Extend the course code to two knobs: `predict(x) = w·x + b`, trained on
   the two points `(1, 5)` and `(2, 7)` with the loss averaged over both. Derive
   `∂L/∂w` and `∂L/∂b` with the chain rule, implement the loop, and check it learns
   `w = 2, b = 3`. You have just written full-batch gradient descent over a gradient
   *vector*.
4. **(medium)** Write a Zig program that calls `makeSpirals`'s logic (copy the 19 lines)
   and prints the 400 points as `x y label` rows; plot them with any tool you like.
   Then predict: what would the plot of a *linear* classifier's best decision boundary
   look like on top of it, and roughly what accuracy could it reach?
5. **(hard)** Build the strongest linear classifier you can on spirals data, in pure
   Zig: `logits = W·x + b` with `W: [2][2]f32`, softmax + cross-entropy loss (§2.7),
   hand-derived gradients (for softmax+CE the per-logit gradient is the pleasingly
   simple `p − onehot(true class)` — derive or look it up), full-batch descent. Train
   until the loss plateaus and report accuracy. It will plateau far from 100% — no
   line can do better — and *that* wall is why §2.6's hidden layers exist. Keep your
   program: in Chapter 8 you will solve the same task properly in a dozen lines.

---

[Previous: Just enough Zig](01-just-enough-zig.md) · [Next: Tensors from scratch](03-tensors-from-scratch.md)
