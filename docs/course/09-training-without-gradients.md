# Chapter 09 — Training without gradients: evolution strategies

*Part III — Learning*

[Chapter 8](08-training.md) built the full backpropagation stack: an autograd graph kept alive by exec scopes, optimizers with per-parameter state, clipping, schedules, bit-exact checkpoints. Every piece of it rests on one assumption so basic it was never stated: *the loss is differentiable with respect to every parameter, and you can afford to compute and store what the chain rule needs*.

This chapter removes that assumption. `fucina.es` (`src/es.zig`) trains models using **forward passes only** — no `backward()`, no graph, no optimizer state, no gradient anywhere in the process. The signal is a single scalar *reward* per candidate model, and a reward can be anything you can compute: a negative loss, a classification accuracy, a regex match on generated text, a rule that checks whether a reply starts with "Ahoy!". The method is **evolution strategies** (ES), and the star of the chapter is the trick that makes it practical at scale: *noise that is never stored, only regenerated from seeds*.

The honest framing up front, because this chapter will spend real effort on an algorithm with a real weakness: ES is not a drop-in replacement for backprop. `docs/TRAINING.md` §13 says it plainly — "ES needs MANY more iterations than backprop for the same movement". What ES buys instead is *reach*: it trains things gradients cannot touch, in memory backprop cannot fit, with parallelism backprop cannot exploit as cheaply.

## 9.1 Learning from rewards alone

Here is the entire algorithm, quoted from the module doc of `src/es.zig:11-14`:

```text
eps_n ~ N(0, I)                                n = 1..population
R_n   = reward(theta + sigma * eps_n)          (forward passes only)
C_n   = (R_n - mean(R)) / (std(R) + 1e-8)      (z-score, ddof = 0)
theta += (alpha / population) * sum_n C_n * eps_n
```

In words: take your parameter vector θ. Draw a *population* of Gaussian noise vectors ε₁…ε_N. For each member n, evaluate the model at θ + σ·εₙ — a plain forward pass producing a scalar reward Rₙ. Normalize the rewards (z-score: subtract the mean, divide by the standard deviation). Then step θ in the direction of the reward-weighted average of the noise: perturbations that scored above the population mean pull θ toward themselves, perturbations that scored below push it away.

That is the whole thing. No derivative of the reward is ever taken — the reward function can be discontinuous, discrete, or a black box.

> **ML note** — Why does this converge to anything? Because the update is a Monte-Carlo estimate of a gradient after all — just not the gradient of your reward. Define the *smoothed* objective J_σ(θ) = 𝔼_ε[R(θ + σε)]. A classic identity (the score-function/REINFORCE trick) gives ∇J_σ(θ) = (1/σ)·𝔼[R(θ + σε)·ε]: you can estimate the gradient of the smoothed landscape by sampling noise and weighting it by reward, never differentiating R. ES is therefore best understood as gradient *estimation* by randomized finite differences, with σ controlling how much the landscape is blurred. The estimate's variance scales with the parameter dimension — which is precisely why ES needs large populations and many iterations, and why Chapter 8's exact gradients win whenever they are available.

Two provenance facts, stated precisely because they matter for both science and licensing:

1. **The algorithm is a faithful reimplementation of a specific paper**: "Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning" (arXiv:2509.24372), whose reference implementation lives at `github.com/VsonicV/es-at-scale`. The module doc (`src/es.zig:1-6`) is explicit: the algorithm is "reimplemented here from the paper, not ported: the reference code is under a noncommercial Academic Public License". `docs/TRAINING.md` §13 repeats the stance: the reference "is compared against, never ported into the tree". Fucina wrote the code from the paper's math, then *verified* its algebra against the reference's behavior (§9.7) — a clean-room discipline you will want whenever a useful reference carries a restrictive license.

2. **The reference's simplifications are kept, faithfully.** This is deliberately *vanilla* OpenAI-ES with everything the paper stripped left out by default: no antithetic pairs, no rank shaping, no optimizer state — and, the reference's one documented deviation from the textbook update, **no division by σ**: the 1/σ factor from the ML note above is folded into α, whose default is σ/2. The `es.Config` defaults are the reference's defaults: `sigma = 0.001`, `alpha = null` (meaning σ/2), `population = 30` (`src/es.zig:195-245`). The classic OpenAI-ES improvements you may have heard of (Salimans et al. 2017) — mirrored sampling, centered ranks — the very ones the paper stripped — exist in Fucina too, but strictly opt-in (§9.5), so a default-config run is comparable with the paper.

## 9.2 The trick: noise that is never stored

Now look at the algorithm's memory bill as written. Each εₙ has the same shape as θ. The update at the bottom needs *all N of them* after the rewards come back. Stored naively, that is N × P floats: for the paper's population of 30 on a 0.6-billion-parameter model, 30 × 0.6e9 × 4 bytes ≈ **72 GB of noise** — for one iteration. (That is arithmetic, not a benchmark.) The naive algorithm is unusable at exactly the scale where its gradient-free property becomes interesting.

The fix — the reference's headline trick, and the reason this chapter exists — is that the noise never needs to *exist* as data. Quoting `docs/TRAINING.md:844-848`:

> Noise is never stored: a member's perturbation is a pure function of (seed, iteration, member, slot, element) through the counter-based gaussian, so `perturb`, `restore`, and `update` regenerate it on the fly — O(1) memory beyond the parameters, the reference's headline trick.

You met Fucina's counter-based RNG in [Chapter 8](08-training.md) as a checkpoint contract for dropout masks and APOLLO projections. Here it graduates from convenience to load-bearing architecture. `rng.at(seed, i)` returns the i-th value of a splitmix64 stream in O(1) — no state to advance, no sequence to replay. Stack the derivations and *element j of member n's noise at iteration t* becomes an addressable pure function. From `src/es.zig:545-553`:

```zig
    /// The seed of population member `member` at the CURRENT iteration — a
    /// pure function of (config.seed, iteration, member). Checkpoint
    /// contract; see the module doc.
    pub fn memberSeed(self: *const Self, member: usize) u64 {
        return rng.at(
            self.config.seed ^ seed_domain,
            self.iteration *% @as(u64, self.config.population) +% @as(u64, member),
        );
    }
```

and each registered parameter buffer (a *slot*) gets its stream from the member seed (`src/es.zig:566-572`):

```zig
    /// The noise-stream seed of (member, slot) under the configured scheme.
    fn slotStreamSeed(self: *const Self, member_seed: u64, slot_index: usize) u64 {
        return switch (self.config.noise) {
            .iid => rng.at(member_seed ^ noise_domain, slot_index),
            .correlated => member_seed,
        };
    }
```

> **Zig note** — `seed_domain` and `noise_domain` are *domain separators*: constants XORed into the seed so that different consumers of the same base seed can never collide stream-for-stream. Fucina spells them as hex-encoded ASCII — `const seed_domain: u64 = 0x65735f7365656473; // "es_seeds"` (`src/es.zig:259-260`) — so a hex dump of a derivation is self-describing. The wrapping operators `*%` and `+%` are Zig's explicit two's-complement arithmetic: overflow is *defined* to wrap, rather than being UB (plain `*`/`+` on integers trap on overflow in safe builds). For seed mixing, wrapping is exactly what you want, and the source says so at every call site.

Three consequences, in increasing order of importance:

**Perturbation and restoration are both regeneration.** `perturb(ctx, member)` adds σ·ε to the registered buffers in place; `restore(ctx, member)` regenerates *the same* ε and subtracts it (`restore_mode = .regenerate`, the default). Nothing of size O(P) was allocated in between. The undo is exact up to the classic `(x + t) - t` floating-point drift per element; if you need bitwise restoration — for example to keep an evaluation loop bit-stable over thousands of iterations — `restore_mode = .snapshot` memcpys the parameter bytes back instead, at the cost of exactly one extra copy of the parameters (`src/es.zig:145-152`).

**The update regenerates everything a third time.** The weighted sum Σ Cₙ·εₙ walks every member's stream again, element by element. Count the traffic: at population 8, one iteration regenerates the full noise ~24 times (8 perturbs + 8 restores + 8 update terms). This is why `docs/TRAINING.md:848-854` calls regeneration "the dominant cost of full-parameter ES on CPU" — and why `src/es.zig` draws its noise through `rng.gaussianFillAtFast` (`src/rng.zig:97-144`), a hand-vectorized Box–Muller that replaces the f64 libm transcendentals with f32 polynomial ln/sin/cos evaluated 8 pairs at a time in `@Vector` lanes. Careful: that makes it a *different* (seed → values) mapping than the scalar `gaussianFillAt` — agreeing to a few ulps, but not bitwise — so the two functions are documented as **separate checkpoint contracts**: ES noise lives on the fast mapping, APOLLO and dropout stay on the scalar one, and neither mapping may ever change once checkpoints exist.

**A checkpoint is just an iteration counter.** Since the entire population is a pure function of `(config.seed, iteration)` plus the scheme knobs, resuming a run needs no noise, no member list, no optimizer state file — the `TrainerState.es_*` fields (`src/training_checkpoint.zig`) record sigma, alpha, population, the scheme flags, and `es_iteration`, and there is *no* `optimizer.fucina` in an ES checkpoint at all. The flip side is that the knobs are load-bearing: changing `population`, `seed`, `noise`, or `antithetic` mid-run silently breaks the (seed, iteration) → noise mapping, "exactly like editing an optimizer checkpoint" (`docs/TRAINING.md:1001-1003`).

Contrast this with Chapter 8's memory story. Backprop training holds: the activations of the forward pass (the graph the backward walks), the gradients (one f32 per parameter), and the optimizer state (two more per parameter for AdamW). ES holds: θ, a rewards array of `population` floats, and the forward pass's own working set — with the *inference* lifetime rules, because nothing needs to survive for a backward that never comes. Every O(P) allocation beyond θ is opt-in: the `.snapshot` restore slab, the AWD anchor (§9.5), the `cache_streams` RAM cache for the replay-heavy small-model regime.

## 9.3 Build it yourself: evolution strategies in sixty lines

Before meeting the real trainer, build the whole mechanism from nothing — counter RNG, seed-regenerated noise, three regenerations per iteration — in a single test you can run today. This is **course code** (not from the repo; simplified: it keeps only the cosine half of each Box–Muller pair, and its seed derivation is cruder than `src/es.zig`'s domain-separated one). It compiles and passes with Zig 0.16.0 via `zig test es_tiny.zig`:

```zig
// Course code — a complete evolution-strategies trainer in ~60 lines.
// The point to internalize: noise is REGENERATED from its coordinates
// (seed, iteration, member, element), never stored.
const std = @import("std");

fn splitmix64(state: *u64) u64 {
    state.* +%= 0x9e3779b97f4a7c15;
    var z = state.*;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/// The i-th value of the stream started at `seed`, in O(1): counter-based
/// access is what makes noise a pure function of its coordinates.
fn at(seed: u64, i: u64) u64 {
    var state = seed +% i *% 0x9e3779b97f4a7c15;
    return splitmix64(&state);
}

/// Standard-normal draw `j` of population member `member` at iteration
/// `iter` — the whole trick in one function: same inputs, same value,
/// forever. (Box–Muller; we keep only the cosine half for brevity — the
/// real rng.zig consumes both halves of each pair.)
fn noise(seed: u64, iter: u64, member: u64, j: u64) f32 {
    const member_seed = at(seed, iter *% 1_000_003 +% member);
    const a = at(member_seed, 2 * j);
    const b = at(member_seed, 2 * j + 1);
    const ua = (@as(f64, @floatFromInt(a >> 11)) + 1) * 0x1.0p-53; // (0, 1]
    const ub = @as(f64, @floatFromInt(b >> 11)) * 0x1.0p-53; // [0, 1)
    const r = @sqrt(-2.0 * @log(ua));
    return @floatCast(r * @cos(2.0 * std.math.pi * ub));
}

test "ES from scratch: seed-regenerated noise trains a quadratic" {
    const seed: u64 = 42;
    const dim = 8;
    const population = 16;
    const sigma: f32 = 0.05;
    const alpha: f32 = 0.025; // sigma/2 — the reference's default coupling

    // theta starts far from the optimum (the origin).
    var theta = [dim]f32{ 1.5, -2.0, 0.7, 3.0, -1.2, 0.4, -0.9, 2.2 };

    const lossOf = struct {
        fn of(t: *const [dim]f32) f32 {
            var s: f32 = 0;
            for (t) |v| s += v * v;
            return s;
        }
    }.of;

    const before = lossOf(&theta);
    var rewards: [population]f32 = undefined;

    for (0..2000) |iter| {
        for (&rewards, 0..) |*r, member| {
            // Perturb IN PLACE: theta += sigma * eps   (noise fill #1)
            for (&theta, 0..) |*t, j| t.* += sigma * noise(seed, iter, member, j);
            r.* = -lossOf(&theta); // reward = any forward-pass score
            // Restore: subtract the SAME regenerated noise (fill #2)
            for (&theta, 0..) |*t, j| t.* -= sigma * noise(seed, iter, member, j);
        }
        // z-score the rewards (population statistics, ddof = 0).
        var mean: f64 = 0;
        for (rewards) |r| mean += r;
        mean /= population;
        var varsum: f64 = 0;
        for (rewards) |r| varsum += (r - mean) * (r - mean);
        const stddev = @sqrt(varsum / population);
        // Update: regenerate every member's noise a THIRD time.
        for (&theta, 0..) |*t, j| {
            var acc: f32 = 0;
            for (rewards, 0..) |r, member| {
                const c: f32 = @floatCast((r - mean) / (stddev + 1e-8));
                acc += c * noise(seed, iter, member, j);
            }
            t.* += alpha / population * acc;
        }
    }

    // Thousands of forward passes, zero gradients, zero stored noise:
    // the loss must have collapsed.
    try std.testing.expect(lossOf(&theta) < 0.01 * before);
}
```

Read the loop body and notice what is *absent*: there is no array of noise vectors anywhere. The three marked fills call the same pure function with the same coordinates and get the same values. Also notice what the 2000-iteration count is telling you about an 8-dimensional quadratic — the smoothest, most gradient-friendly objective imaginable. One analytic gradient step of the right size solves this problem *exactly*. That gap is the price of never computing a derivative, and it only widens with dimension.

## 9.4 The real thing: `fucina.es.Trainer`

The production trainer keeps that loop's shape and industrializes every line of it. The surface, as the usage sketch from `docs/TRAINING.md` §13 shows it (the API lives in `src/es.zig`; today's code, not a frozen contract — the whole public API is pre-1.0):

```zig
var trainer = try fucina.es.Trainer.init(allocator, .{ .sigma = 0.01, .population = 16 });
defer trainer.deinit();
try trainer.addParam(&w);
```

**Registration takes anything with float storage.** `addParam`/`addParamNamed` accept a pointer to any f32/f16/bf16 facade tensor — autograd variable, frozen constant, or typed 16-bit tensor; ES treats them identically because it never needs a `GradState` (docs/REFERENCE.md §11). Let that sink in after Chapter 8's careful `requiresGrad` bookkeeping: *frozen constants are first-class trainable parameters here*. `addRegistry` registers an entire `ParamRegistry` in one call — trainable and frozen entries alike — deduplicating tied storage the way torch's `named_parameters()` does. Duplicate registration errors with `DuplicateParam`; non-contiguous views with `NonContiguousParam`.

The smallest complete run is the machine-verified snippet from `docs/REFERENCE.md` §11 (the ES section), which trains a bare 4-element constant to shrink the sphere objective:

```zig
test "ES: perturb/evaluate/restore/update shrinks the sphere objective" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // ES needs no gradients: a plain constant is a first-class parameter.
    var theta = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{4}, &.{ 1, -2, 0.5, 3 });
    defer theta.deinit();

    var trainer = try fucina.es.Trainer.init(alloc, .{ .sigma = 0.05, .population = 8, .seed = 7 });
    defer trainer.deinit();
    try trainer.addParamNamed(&theta, "theta");

    const sumSq = struct {
        fn of(t: anytype) !f32 {
            var s: f32 = 0;
            for (try t.dataConst()) |v| s += v * v;
            return s;
        }
    }.of;

    const before = try sumSq(&theta);
    var rewards: [8]f32 = undefined;
    for (0..200) |_| {
        for (&rewards, 0..) |*r, member| {
            try trainer.perturb(&ctx, member); // theta += sigma * eps_member
            r.* = -(try sumSq(&theta)); // reward from a plain forward pass
            try trainer.restore(&ctx, member);
        }
        _ = try trainer.update(&ctx, &rewards); // z-score + fp32-accumulated step
    }
    try std.testing.expect(try sumSq(&theta) < before);
    try std.testing.expectEqual(@as(u64, 200), trainer.iteration);
}
```

**Two evaluation shapes** cover the two regimes ES lives in (`docs/TRAINING.md` §13):

- **In place**, for big models: `perturb`/`restore` mutate the registered buffers directly, so *one* model instance serves the whole population, each forward saturating all cores. Exactly one member may be active at a time — a second `perturb` before `restore` errors with `MemberActive`, a deliberate tripwire against the classic bug of evaluating one member on top of another. `step(ctx, evaluator)` packages the whole iteration (perturb → `evaluator.eval(member)` → restore per member, then one `update`), and restores before propagating if an evaluation fails mid-population.
- **Member-parallel replicas**, for small parameter sets (adapters, MLPs): `materializeMember(member, dst)` writes θ + σ·εₙ into caller-owned replica buffers *without touching shared θ*, and `evaluateMembers(evaluator, rewards, workers)` fans the population out over OS threads that pull members from a shared counter and call `evaluator.evalMember(worker, member)`. This is ES's celebrated "embarrassingly parallel" face: the members are independent by construction, and the only data crossing threads is one scalar reward each. (In the paper's setting, that is what lets the algorithm scale across *machines* — the coordination traffic per iteration is a seed and a float per member, not gradients.)

> **Zig note** — Both evaluators are *duck-typed*: `step(self, ctx, evaluator: anytype)` requires only that `evaluator.eval(member)` (or `evalMember(worker, member)`) exists and returns `!f32`. No interface declaration, no vtable, no closure type — the comptime type system checks the method's existence and signature at the call site, and the call is statically dispatched. Compare `AnyOptimizer` in Chapter 8, where dynamic dispatch was genuinely needed and a hand-rolled vtable was built; here monomorphization suffices, so nothing is built.

**The update is engineered like an optimizer step, minus the state.** `update(ctx, rewards)` computes reward statistics in f64 (ddof 0, sequential summation), shapes the coefficients (§9.5), then applies the weighted noise sum chunk-parallel across the worker team with fp32 accumulation and *pinned rounding placement* — explicit mul-then-add, no FMA contraction — narrowing once to each parameter's dtype (docs/REFERENCE.md §11). Because every element's noise is O(1)-addressable, the parallel kernels are element-independent maps, "bitwise identical to the serial loop for ANY thread count" — the same determinism discipline as `optim.zig`'s `parallelMap` from Chapter 8. And `update` is *mutate-last*: every fallible step (allocations, coefficient shaping, ternary vote collection) runs before the first parameter byte changes, so a failed update is a no-op rather than a corrupted θ (`src/es.zig:809-822`).

One more Chapter-8 echo worth naming: dropout regenerated masks from seeds, APOLLO regenerated projection matrices from seeds, the data loader regenerated permutations from seeds — and ES regenerates its entire *population* from seeds. The pattern has fully inverted the usual relationship between randomness and reproducibility: in this codebase, random numbers are the most rigorously versioned artifacts of all, because a checkpoint's meaning depends on them.

## 9.5 Antithetic pairs, reward shaping, and the drift brake

Three opt-in extensions sit clearly outside the reference algorithm (kept opt-in *so that* a default run reproduces the paper). Each earns its place with a distinct failure mode it addresses.

**Antithetic (mirrored) sampling** — `antithetic = true`, Salimans et al. 2017 — pairs members as (+ε, −ε) sharing one noise draw: member 2k draws the stream, member 2k+1 applies its negation. The variance-reduction intuition: evaluating both directions of the same perturbation cancels the "luck of the draw" — if the pair scores (good, bad) you have real signal about that direction; (good, good) says the direction is irrelevant and the fold cancels it. The implementation gets an efficiency bonus for free, from the comment inside `update` (`src/es.zig:848-853`):

```zig
        // Antithetic pairs fold to one stream each: (+eps, -eps) with
        // coefficients C_2k and C_2k+1 contribute (C_2k - C_2k+1) * eps_k —
        // the OpenAI-starter pair-difference form — halving update-side
        // noise regeneration. Division stays by the FULL population (the
        // starter's `g /= returns_n2.size`).
        const n_streams = if (self.config.antithetic) rewards.len / 2 else rewards.len;
```

It requires an even population, and — like everything that shapes noise — it is part of the checkpoint contract (`es_antithetic`).

**Reward shaping** is the stability lever, and `docs/TRAINING.md` §13 devotes its sharpest analysis to it. The reference's z-score is *affine*, so it preserves outlier magnitude: with an unbounded reward like raw −CE, "one catastrophically-perturbed member can carry several times the coefficient of every well-behaved member" (`docs/TRAINING.md:918-919`), degenerating the update into "flee the bad direction" — the documented signature is descent, then growing population variance, then collapse. The alternative, `reward_norm = .centered_ranks`, is Salimans-style fitness shaping: sort members by reward, map rank to the fixed grid rank/(N−1) − 0.5 in [−0.5, 0.5]. It is invariant to any monotone transform of the reward and immune to outliers by construction — the worst member is exactly −0.5 no matter *how* bad. The implementation is five lines plus one deliberate subtlety (`src/es.zig:831-839`):

```zig
                // Ascending by (reward, member index): the index tiebreak
                // makes the comparator a total order, so the sort result —
                // and therefore the coefficients — are deterministic.
                std.mem.sort(usize, order, rewards, struct {
                    fn lessThan(r: []const f32, a: usize, b: usize) bool {
                        if (r[a] != r[b]) return r[a] < r[b];
                        return a < b;
                    }
                }.lessThan);
```

Even sort-tie order is a checkpoint contract here (numpy's argsort tie order is implementation-defined; this one is pinned).

But shaping is a genuine trade, not an upgrade — Fucina measured the reversal. On the smooth spirals task with a bounded, well-behaved reward, "z-score converges and `centered_ranks` stalls — rank shaping trades magnitude information for outlier immunity, so pick by reward regime, not by habit" (`docs/TRAINING.md:992-994`). Ranks also have a failure z-score doesn't: on all-equal rewards, z-score produces all-zero coefficients and self-stops exactly, while ranks still emit their full ±0.5 spread — a mean-zero phantom step. Bounded reward → z-score; unbounded reward → centered ranks plus checkpoint selection. Neither normalization shrinks the step near a sharp optimum (both rescale to a fixed spread every iteration), so the practical brakes remain σ, saturating rewards, and eval-selected checkpoints.

**Anchored weight decay** (`anchor_decay = .l1|.l2` + `anchor_lambda`; arXiv:2605.30148, the same group's anti-forgetting follow-up) addresses the slow pathology: because each ES step is a noisy estimate, θ accumulates a random walk in reward-irrelevant directions, drifting away from the pretrained model it started from. After each update, a proximal step pulls θ toward a fixed anchor captured by `captureAnchor()` — l2 shrinks θ − θ₀ by (1 − αλ), l1 soft-thresholds it at αλ, exactly zeroing small deltas and making the fine-tuning delta sparse. It is fine-tuning-only (anchoring a random init pins the model to noise), and it carries the chapter's most treacherous resume rule: the anchor is *never serialized*, so on resume you must capture it from the reconstructed initial weights **before** loading the checkpoint — otherwise you anchor to the middle of the run.

## 9.6 Walking `es_spirals`

`examples/es_spirals/main.zig` is the from-scratch acceptance test of the whole subsystem, and its header states why "from scratch" is the point: "fine-tuning starts at a good solution, so only a from-scratch run proves the method optimizes rather than merely perturbs" (`examples/es_spirals/main.zig:3-5`). Same task, same two-hidden-layer tanh MLP, same data generator as Chapter 8's `spirals.zig` — only the learning signal differs.

The model declaration carries the chapter's thesis in its doc comment (`examples/es_spirals/main.zig:39-48`):

```zig
/// spirals.zig's MLP, as CONSTANTS: ES needs no gradients, so nothing here
/// is a variable — the same struct doubles as the trainable canonical model
/// and as per-worker replicas.
const Model = struct {
    w1: Tensor(.{ .h1, .in }),
    b1: Tensor(.{.h1}),
    w2: Tensor(.{ .h2, .h1 }),
    b2: Tensor(.{.h2}),
    w3: Tensor(.{ .class, .h2 }),
    b3: Tensor(.{.class}),
```

Every tensor Chapter 8 made a gradient-tracked variable is a plain constant here — 4,482 parameters (64-wide hidden layers), zero `GradState`s. The forward is `spirals.zig`'s forward *verbatim*, and the reward is mean cross-entropy on the full batch, negated and computed inside an ordinary inference-style scope (`examples/es_spirals/main.zig:146-153`):

```zig
/// Mean CE over the full batch (scoped, forward only).
fn meanCe(ctx: *ExecContext, model: *const Model, x: *const Tensor(.{ .batch, .in }), labels: []const usize) !f32 {
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const logits = try forwardLogits(ctx, model, x);
    const loss = try logits.crossEntropy(ctx, .class, labels);
    return loss.item();
}
```

No lifetime ceremony, no `backward`, no Chapter-8 rule about keeping the graph alive — there is no graph to keep alive.

This example runs the **member-parallel** shape: 4,482 parameters replicate for pennies, so each of the (default four) worker threads owns a complete replica — model, `ExecContext`, batch tensor — and only scalar rewards cross threads (`examples/es_spirals/main.zig:205-216`):

```zig
const Evaluator = struct {
    trainer: *const es.Trainer,
    workers: []Worker,
    labels: []const usize,

    pub fn evalMember(self: *const Evaluator, worker: usize, member: usize) !f32 {
        const w = &self.workers[worker];
        var views = w.model.byteViews();
        try self.trainer.materializeMember(member, &views);
        return -(try meanCe(&w.ctx, &w.model, &w.x, self.labels));
    }
};
```

`byteViews()` returns the six parameter buffers *in the canonical registration order* — `materializeMember`'s destination contract is positional (`dst[k]` is slot k, byte lengths validated with `ReplicaShapeMismatch`), so the example pins that order in `byteViews`'s doc contract, which `register` is documented to share (`examples/es_spirals/main.zig:111-112`) — two functions pinned to each other. After all the machinery, the training loop is two lines (`examples/es_spirals/main.zig:323-325`):

```zig
    for (0..iterations) |iter_i| {
        try trainer.evaluateMembers(&evaluator, rewards, workers);
        _ = try trainer.update(&ctx, rewards);
```

The run's config departs from the `es.Config` defaults deliberately — from-scratch optimization is a different regime than fine-tuning: `sigma = 0.1`, `alpha = 0.1`, `population = 128`, antithetic whenever the population is even, z-score shaping (the reversal from §9.5 is *why* z-score is this demo's default). And it is **self-verifying**: the process exits non-zero unless the trained network reaches `--target` accuracy (default 0.90) on the training set — chance is 0.50, so a pass demonstrates genuine from-scratch learning every time it runs. Run it yourself:

```
zig build es-spirals -Doptimize=ReleaseFast
```

The documented result (`docs/TRAINING.md:986-987`; the machine is named in the example header, `examples/es_spirals/main.zig:10` — M1 Max, ReleaseFast; a dated, machine-specific snapshot like every number in this course): "100% accuracy / CE -> 0 in ~15k iterations (~75 s ReleaseFast, ~5 ms/iteration), population 128". Sit with that number for a moment. Fifteen *thousand* iterations, each evaluating 128 candidate models, for a task Chapter 8's `spirals.zig` trains in 2,000 gradient steps (`examples/spirals/main.zig:26`). Nearly two million member evaluations where backprop needed two thousand forward-backward passes. On a 4,482-parameter model the wall-clock is 75 seconds and nobody cares; the *ratio* is what you must carry to §9.8, where parameters number in the billions.

## 9.7 Compared against, never ported: the parity story

Chapter 8 pinned every optimizer to its reference implementation with golden tests. ES gets the same religion, with an extra constraint: the reference code cannot be copied, so parity has to be established across a clean-room boundary. Three layers (`docs/TRAINING.md` §13, "Parity evidence"):

1. **Bit-level RNG goldens.** `tools/gen_es_goldens.py` replicates the repo's splitmix64 integer stream bit-level in numpy, computes the gaussian with f64 libm (the scalar `gaussianFillAt` mapping), and re-derives the update algebra, producing goldens for `src/es_tests.zig`. These are *tolerance* goldens — not from any algebraic slack, but because es.zig draws through the `gaussianFillAtFast` polynomial mapping of §9.2, a few f32 ulps from the generator's f64 libm values (with libm's own ulp variation across platforms on top).
2. **Serial-vs-parallel, bitwise.** A test-local straight-line serial reference pins the chunk-parallel kernels *bitwise* — the "any thread count" determinism claim of §9.4 is enforced, not asserted.
3. **The actual reference, on identical noise.** `tools/check_es_parity.py` runs the real es-at-scale code — perturb, restore, z-score, update, imported from a local clone of `refs/es-at-scale`, with a stub model runner so no vLLM is needed — against a *torch transcription of es.zig's algebra*, feeding both the same noise. The result is bitwise `torch.equal` on f32, f16, and bf16, under both noise schemes. The one deliberate substitution is the RNG itself: repo-owned splitmix64 instead of torch's Philox, because the RNG is a Fucina checkpoint contract (§9.2). The AWD kernels get the same treatment against the actual `es-awd` reference — bitwise for l1 and l2 — and the checker even documents a one-ulp discrepancy *between the two references* (es-awd's fused `add_(alpha=)` vs es-at-scale's explicit mul-then-add; es.zig follows es-at-scale).

Note the shape of layer 3 carefully, because the licensing stance depends on it: the Academic-Public-Licensed code is executed *as an oracle* in an out-of-tree checker; what lives in the tree was written from the paper. Faithfulness goes deep enough to preserve the reference's *quirks* — the `.correlated` noise scheme exists solely because the reference library reseeds one generator per tensor, so same-length tensors get identical noise: "an acknowledged artifact its authors kept; the paper's results used it" (`docs/TRAINING.md:889-890`). Reproducing a paper means reproducing its artifacts, flagged as artifacts, or you can no longer compare results.

## 9.8 Where gradients cannot go

Everything so far, ordinary backprop could also do — slower memory bill, but it could. The reason ES is *first-class* in Fucina rather than a demo is the territory where gradients are structurally unavailable:

**Quantized and ternary weights.** A gradient is a statement about infinitesimal change, and a weight packed into a 2-bit trit has no infinitesimal neighborhood — the gradient world's answer is the straight-through estimator, a controlled lie you will meet in [Chapter 14](14-the-low-bit-frontier.md). ES needs no lie: `es.Trainer` registers packed TQ2_0 ternary genomes directly (`addTernaryParam`), perturbs them by sparse seed-regenerated trit flips, and updates by fitness-weighted vote-and-threshold — the weights never leave {−1, 0, +1}·d, "members evaluate through the real TQ2_0 int8 kernels and the trained state is byte-for-byte the served model" (`docs/TRAINING.md:1018-1019`). Training *is* the packed inference model. The full design record is `docs/TERNARY.md` and its acceptance demo is `zig build es-ternary-spirals`; Chapter 14 tells that story properly. (One boundary stays honest even here: `es-finetune --mode full` on a quantized *GGUF* is rejected up front with `QuantizedWeightsUnsupported` — gaussian noise cannot be added to K-quant blocks; ternary genomes work because ES perturbs them in their own discrete language, not with gaussians.)

**Non-differentiable rewards.** ES composes with anything you can score from a forward pass. `examples/es_finetune/main.zig` ships a DeepSeek-R1-style `rule` reward on greedy generations — unigram-F1 against the gold answer plus 0.1 for matching a response envelope ("starts with `Ahoy!`, ends with `matey.`"). There is no gradient of "the reply matched a regex" through a sampling loop; there doesn't need to be. This is the paper's actual pitch — fine-tuning LLMs on rule rewards *beyond* reinforcement learning — and it is why the algorithm's signal type (one scalar per candidate) is a feature, not a poverty.

**Memory and parallelism at LLM scale.** `zig build es-finetune` is `finetune.zig`'s gradient-free twin — deliberately the same dataset plumbing, the same trainer forward, the same checkpoint layout, so the backprop and ES runs compare apples-to-apples. `--mode lora` perturbs only the LoRA adapters; `--mode full` perturbs every resident float weight of the base model — the paper's full-parameter setting, with no backward memory at any model size. That walkthrough belongs to [Chapter 15](15-training-llms-on-cpu.md), after GGUF loading and LoRA are on the table; what belongs here is its cost sheet, quoted with its conditions (`docs/TRAINING.md:959-965`, M1 Max, Qwen3-0.6B, ReleaseFast): "lora + nll ≈ 0.9 s per member-eval (population 8 x batch 5 ≈ 7-10 s/iteration); full mode adds the noise-regeneration cost over 0.6 B parameters (≈ 30 s/iteration at population 4, batch 1); rule rewards are generation-bound. ES needs MANY more iterations than backprop for the same movement — the paper runs 300-500 iterations at population 30 — so treat the demo defaults as a mechanism showcase, not a convergence recipe." And when you get there, §15.10 widens the frame: `es-finetune` is one road of a four-way post-training menu — gradients, evolution, trit-planes, or a trained context — the one that asks only for a scalar per candidate.

That final sentence is this chapter's required reading. ES does not match gradient training's sample efficiency and Fucina never claims it does; what the repo claims — and proves with self-verifying demos and bitwise parity — is that the *mechanism* is correct, deterministic, resumable, and reaches places backprop cannot.

(A footnote for GPU builds: on `-Dgpu=cuda`, slots whose storage is GPU-resident run perturb/restore/update/anchor as device kernels that are bitwise identical to the CPU kernels for any launch geometry — so even checkpoints are device-independent; `docs/TRAINING.md` §13.)

The documented pitfalls, condensed (`docs/TRAINING.md` §13):

- Perturb without restore → the next perturb errors with `MemberActive`. The tripwire is your friend; don't catch it.
- One NaN/Inf reward poisons the z-score for the entire iteration — clamp or zero pathological rewards in the evaluator (the reference scores failed rollouts 0.0).
- `population`, `seed`, `noise`, `antithetic`, and the ternary knobs are all part of the noise checkpoint contract; changing any of them mid-run silently invalidates a resume.
- With AWD, capture the anchor from the initial weights *before* loading a checkpoint on resume.

## What you now know

- Evolution strategies train with forward passes only: perturb θ with Gaussian noise, score each member with any scalar reward, take a fitness-weighted step — a Monte-Carlo estimate of the *smoothed* objective's gradient, no derivative of the reward ever taken.
- Fucina's `fucina.es` reimplements ES-at-scale (arXiv:2509.24372) *from the paper* — the Academic-Public-Licensed reference code is compared against, never ported — keeping the reference's simplifications (no 1/σ in the update, α = σ/2, plain z-score) by default and offering antithetic pairs, centered ranks, and anchored weight decay strictly opt-in.
- The scale trick: noise is a pure function of (seed, iteration, member, slot, element) through a counter-based RNG, so `perturb`, `restore`, and `update` regenerate it on the fly — O(1) memory beyond the parameters, versus N×P stored noise naively and versus backprop's graph + gradients + optimizer state.
- The same trick makes checkpoints trivial (resume = an iteration counter; no optimizer state file) and makes the RNG mapping itself a versioned contract — down to sort-tie order in the rank shaping.
- Two evaluation shapes: in-place perturb/restore for big models (one instance, `MemberActive` tripwire, mutate-last update), and member-parallel replicas for small parameter sets (`materializeMember` + `evaluateMembers`, only scalar rewards cross threads).
- Reward shaping is a regime choice, not a default: z-score preserves magnitude but lets one catastrophic member dominate unbounded rewards; centered ranks are outlier-immune but stall where magnitude matters — Fucina measured both directions on the same task.
- ES's parity is proven in three layers, ending with the actual reference code run as an oracle against a transcription of es.zig's algebra — bitwise equal on identical noise, on three dtypes, both noise schemes.
- ES needs many more iterations than backprop for the same movement; its value is reach — quantized/ternary weights, non-differentiable rule rewards, no-backward-memory full-parameter fine-tuning — not efficiency.

## Explore the source

- `src/es.zig` — the whole subsystem in one file; read the module doc first: it is the best short summary of the design, licensing stance, numerics, and checkpoint contract.
- `examples/es_spirals/main.zig` — the member-parallel walkthrough of §9.6; 368 lines you can now read top to bottom.
- `docs/TRAINING.md` §13 — the design record: reward-shaping analysis, AWD, measured costs, pitfalls.
- `docs/REFERENCE.md` §11 (the `fucina.es` section) — the API reference with the machine-verified sphere test.
- `src/es_tests.zig`, `tools/gen_es_goldens.py`, `tools/check_es_parity.py` — the three parity layers of §9.7.
- `src/rng.zig` — `at`, `gaussianFillAt`, `gaussianFillAtFast` and the two-contracts note; the enabling infrastructure of the whole chapter.
- `examples/es_finetune/main.zig` — the LLM fine-tuning twin, ahead of [Chapter 15](15-training-llms-on-cpu.md).

## Exercises

1. **(Easy)** In the course snippet of §9.3, set `population = 2` and rerun. Then restore the population and instead set `sigma = 1.0`. Explain each failure (or slowdown) using §9.1's ML note: what does population size control in a Monte-Carlo gradient estimate, and what does σ do to the smoothed landscape J_σ?
2. **(Easy)** Run `zig build es-spirals -Doptimize=ReleaseFast -- --norm centered_ranks` and compare with the default z-score run. You should reproduce the §9.5 reversal on your machine. Then design a *bounded* variant of the reward (e.g. accuracy instead of −CE) by editing the example's `evalMember`, and check whether the gap narrows.
3. **(Medium)** Add antithetic sampling to the course snippet: members 2k and 2k+1 share one noise stream with opposite signs, and the update folds each pair to `(C_2k − C_2k+1) · eps_k` while still dividing by the full population. Verify the quadratic converges in fewer iterations at the same population, and confirm your fold regenerates each pair's noise only once in the update.
4. **(Medium)** The course snippet's member-seed derivation (`iter *% 1_000_003 +% member`) is cruder than the repo's `iteration *% population +% member` (§9.2). First swap in the repo's derivation, then break the checkpoint contract on purpose: rerun with the same seed but `population` changed from 16 to 17. Explain why *every* member's noise now differs at every iteration after the first — not just the extra member's (at iteration 0 the index `iteration *% population +% member` collapses to `member`, so population only enters from iteration 1 on) — and why this makes `population` part of the noise contract that `TrainerState` must persist.
5. **(Hard)** Implement `.snapshot` restore mode in the course snippet (copy θ before the population loop, memcpy it back after each member) and measure, over many iterations, the largest per-element divergence between a `.regenerate` run and a `.snapshot` run of the same seed. You are measuring the accumulated `(x + t) − t` drift that `src/es.zig:145-152` documents. At what σ does it stop being zero on your machine?

---

[Previous: Training — making the machine learn](08-training.md) · [Next: The guitar amp — real-time neural audio](10-the-guitar-amp.md)
