# TRAINING.md — training with Fucina

How to train models on the eager autograd runtime: the tensor-lifetime rules
(and why they differ from inference), exec scopes, optimizers, param groups,
LR schedules, gradient clipping, loss options, dropout, gradient
checkpointing, checkpoint files, LoRA fine-tuning of GGUF models, gradient
verification, the export loop back to GGUF, and performance. Everything here
is exercised end-to-end by `zig build spirals`
and `zig build finetune` and unit-tested in `src/optim_tests.zig` /
`src/ag/tensor_tests.zig` / `src/ag/checkpoint_tests.zig` /
`src/lora_tests.zig` / `src/llm/qwen3/train_tests.zig` /
`src/llm/qwen3/train_golden_tests.zig`.

## 1. A complete training step

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

## 2. Tensor lifetime: why training is not inference

**Inference habit:** deinit every intermediate as soon as its consumer has run.
Correct, and optimal — released buffers go straight back to the `BufferPool`.

**Training rule:** every tensor on the path from the parameters to the loss
must stay alive until `backward()` returns.

**Why.** Each differentiable op result owns two things with *different*
ownership models:

- its **value** (`RawTensor`) — refcounted storage. Backward functions clone
  *views* of the operand values they need, so early release of a value never
  dangles data; the views keep the storage alive.
- its **GradState** — the autograd graph node. It is **single-owner, not
  refcounted**: the tensor owns it, `tensor.deinit()` destroys it
  unconditionally, and the consumers' backward functions hold **raw
  `*GradState` pointers** to it (see `src/ag/backward.zig`; the scheduler in
  `src/ag/core.zig` walks those pointers).

So "why are those freed?" — `deinit` always frees the node, by design: no
atomic refcount traffic on the eager hot path, no ownership cycles, and
inference (where `grad_state == null`) pays nothing. The price is a rule:
deinit an intermediate before backward and the backward pass walks a dangling
node — undefined behavior, not an error you can catch. Leaf parameters are
unaffected between steps (their GradState persists; only the accumulated
gradient is dropped by `zeroGrad`).

**The implicit mechanism: exec scopes.** The ctx is already threaded through
every op call, so it is the natural owner. While a scope is open, **every
tensor returned by a facade op is owned by the innermost scope** and the value
you receive is a borrow:

```zig
const scope = ctx.openExecScope();
defer ctx.closeExecScope(scope);     // releases everything, newest first
const logits = try forwardLogits(&ctx, &model, &x);   // zero ceremony inside
```

- Scope-owned results carry a `scope_owned` flag and their `deinit` is a
  **safe no-op** (arena-allocator semantics: the owner releases at close).
  So defer-deinit forward code — the inference idiom, including
  `ctx.replace` for residual streams — runs unchanged under a scope: write
  the forward once, train it by opening a scope around it. Tensors you
  create explicitly (`variable`/`constant`/`fromSlice`) and gradients you
  fetch (`grad`/`gradView`) stay yours; never USE a scope-owned borrow after
  the scope closes.
- Scopes nest with stack discipline (a nested eval scope inside a training
  scope releases only its own suffix), and a failed op mid-forward leaks
  nothing — the scope already owns the prefix, so user model code needs no
  errdefer chains at all.
- Adoption is wired into the op tails themselves (`finishOp` in
  `src/ag/tensor.zig`), covering both differentiable results and no-grad
  results (eval on constants, `argmax`, `topK`, the packed-RHS fast paths).
  Typed/quantized-constant tensor methods are not adopted — those are
  weights/caches with explicit lifetimes.
- With no scope open, nothing changes: inference code keeps deinit-ASAP
  semantics and pays one branch per op.
- Close a scope only when no `backward()` over its graph is pending.

(An explicit per-tensor owner, `Tape`, was prototyped early on and removed
unused: scopes cover its eval niche via the no-grad op tails, and nested
scopes release suffixes more finely than its all-or-nothing reset.)

The scope is *not* the inference "frame" helper sketched in `MEMORY-MODEL.md`
§5, and deliberately does not replace deinit-ASAP in the inference engines —
that option was evaluated and rejected; see the note at the end of
MEMORY-MODEL §5.

## 3. Optimizers

All in `fucina.optim` (`src/optim.zig`), each a faithful port of its reference,
verified by golden parity tests against the actual reference implementations
(PyTorch 2.12 / Keller Jordan's muon.py / apollo_torch — see
`src/optim_tests.zig`).

| Optimizer | Reference | State / param elem | Notes |
| --- | --- | --- | --- |
| `SGD` | `torch.optim.SGD` | 0 (4 B with momentum; 2 B with `state_dtype = .bf16`) | momentum buffer = first (decayed) grad, not zeros — in bf16 the clone is stored NARROWED; coupled L2 decay; nesterov needs momentum>0, dampening=0 |
| `AdamW` | `torch.optim.AdamW` | 8 B (6 B with `state_dtype = .bf16`; 4 B with `second_moment_dtype = .bf16` too) | decay before the step; `eps` outside the bias correction |
| `Muon` | Keller Jordan reference | 4 B (2 B with `state_dtype = .bf16`) | rank>=2 params only (conv flattened `[d0, rest]`); Newton-Schulz-5 in f32; `scale = .spectral` (Keller) or `.match_rms_adamw` (Moonlight); built-in AdamW fallback (its own `state_dtype`/`second_moment_dtype` in `config.fallback`) |
| `Apollo` | apollo_torch (arXiv 2412.05270) | ~`8*rank*max(dim)` B moments + `4*rank*min(dim)` B resident projection per matrix | rank-2 params only; `ApolloConfig.mini()` for rank-1 tensor scaling; legacy-HF AdamW fallback (decay *after* the step — deliberate); state stays f32 (no `state_dtype`): the moments are rank-compressed already and the f64 channel-scaling reduction reads them directly |

**bf16 optimizer state (`optim.StateDType`, default `.f32`).** Every
elementwise optimizer can store its moment/momentum buffers in bf16; step math
stays f32 (widen-on-read / narrow-on-write with the NaN-guarded
round-to-nearest-even `dtype.f32ToBf16`), updates remain element-independent
fixed-boundary maps (bitwise identical for any thread count), and the f32
default leaves every existing checkpoint, golden test, and byte-identical v3
frame unchanged. First moments/momentum tolerate bf16 well (per-step relative
change ~5-10% at beta 0.9-0.95). AdamW/Adam `v` is the precision-sensitive one
— with beta2 = 0.999 its ~0.1%/step change sits BELOW bf16's ~2^-8 ≈ 0.39%
resolution, so the EMA can round to a no-op and stall — which is why it has a
separate `second_moment_dtype` opt-in. Memory: full-param AdamW on Qwen3-0.6B
(≈595M params) drops from 4.76 GB state to 3.57 GB (m bf16) / 2.38 GB (m+v
bf16); the default LoRA fine-tune (1.15M adapter params) saves only ~4.6 MB of
its 9.2 MB state — the knob matters at full-parameter/embedding scale, not for
small adapters. The step is memory-bound, so bf16 state also makes it faster
(§11).

**Param routing (Muon/Apollo):** hidden 2D matrices → `addParam`; embeddings
and output/classifier heads → `addFallbackParam` (they are 2D but must not be
orthogonalized/rescaled); 0D/1D params (biases, norms) auto-route to the
fallback. Each variable belongs to exactly one optimizer: duplicates within
one instance are rejected (`error.DuplicateParam`), but registering the same
tensor with two *different* instances (e.g. two OptimizerSet groups) is not
detectable and silently double-steps — cross-instance uniqueness is your
responsibility. Parameters must outlive the optimizer (it holds refcounted
views of their storage plus their GradState pointers).

## 4. Param groups, LR schedules, clipping

**Param groups** are optimizer instances: a PyTorch group is exactly
{hyperparams, params, state}, which is what one instance is. `OptimizerSet`
makes N instances feel like one optimizer:

```zig
var decay = optim.AdamW.init(allocator, .{ .lr = 1e-3, .weight_decay = 0.1 });
var no_decay = optim.AdamW.init(allocator, .{ .lr = 1e-3, .weight_decay = 0 });
// ... addParam matrices to `decay`, biases/norms to `no_decay` ...
var set = optim.OptimizerSet.init(allocator);
try set.add(&decay);
try set.add(&no_decay);
// set.step / set.zeroGrad / set.clipGradNorm / set.saveState / set.loadState
```

Mixing optimizer types in one set works (it is type-erased), e.g. Muon for the
trunk and a separately-tuned AdamW for an adapter.

**LR schedules** attach to the public `config.lr` fields and rescale them from
their captured bases; the factor is a pure function of the step, so resuming
from a checkpoint just re-applies it:

```zig
try sched.attach(&muon.config.lr);
try sched.attach(&muon.fallback.config.lr);   // fallback lr schedules too
sched.apply(optim.warmupCosineFactor(step, total, warmup, min_factor));
```

**Gradient clipping** is `torch.nn.utils.clip_grad_norm_` (global L2 over all
registered params; returns the pre-clip norm). On an `OptimizerSet` it is
global across *all* groups — the PyTorch contract. Order inside a step:
`backward` → `clipGradNorm` → `step` → `zeroGrad`. Note the APOLLO recipes
train with clipping disabled — its norm-growth limiter replaces it.

### Gradient accumulation

The substrate already accumulates: `backward()` ADDS into each param's
persisted `GradState.grad` (`src/ag/core.zig`, `accGradOwnedReady`), leaf
grads live *outside* exec scopes, `step()` consumes them non-destructively,
and `zeroGrad()` is the only clear. Accumulation is one backward per fresh
graph — a repeat over the *same* retained graph fails with
`error.BackwardAlreadyRun` (docs/REFERENCE.md §5.2). A batch of N sequences is therefore N
micro-batch loss+backward passes and ONE optimizer step — the raw-loop recipe
(what `zig build finetune -- --accum-steps N` runs; the LLM trainers' `loss`
is single-sequence, so accumulation IS their batching mechanism):

```zig
for (0..accum_steps) |_| {                        // one scope PER micro-batch
    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    const loss = try trainer.lossExt(&ctx, mb.inputs, mb.labels, .{
        .reduction = .sum,                        // exact token weighting …
        .loss_scale = 1.0 / total_valid,          // … over the window (below)
    });
    try loss.backward(&ctx);                      // ADDS into the leaf grads
    window_loss += try loss.item();               // = true mean CE at window end
}
_ = try set.clipGradNorm(&ctx, 1.0);              // ONCE, after the window
try set.step(&ctx);                               // one MACRO step
set.zeroGrad();                                   // ends the window
sched.apply(optim.warmupCosineFactor(macro_step, ...)); // keyed by MACRO step
```

- **Scopes.** One scope per micro-batch is recommended: each graph is freed
  right after its backward while the leaf grads persist outside the scopes
  (the qwen3/gemma trainers' RoPE caches are trainer-owned, so cross-scope
  backward state stays valid). One big scope around the whole window is also
  legal (`src/llm/qwen3/train_tests.zig` "rope cache: gradient accumulation…"
  is that shape) but retains N graphs in memory until close.
- **Normalization is loss-side and canonical.** Scale the LOSS before
  backward (`lossExt`'s `loss_scale`, or `loss.scale(ctx, f)` on any model).
  The grad-side alternative — `set.scaleGradients(ctx, 1.0/N)` after the N
  backwards, before clip — is deterministic too, but a different fp order:
  NOT bitwise-equal to loss-side scaling. Pick one; tests here pin loss-side.
- **Mean-of-means is not the batch mean.** `.mean` + `loss_scale = 1.0/N`
  equals the true mean over the window ONLY when every micro-batch has the
  same supervised-token count. With `ignore_index` masking (SFT prompt
  masking) counts differ — use `.sum` + `loss_scale = 1.0/total_valid`
  (total non-ignored label positions across the window, counted before the
  window runs); the sum of the scaled losses then IS the true mean CE over
  the window's supervised tokens. This exact arm is what finetune.zig does.
- **Clip ONCE, after the window.** `clipGradNorm` reads the accumulated
  grads; calling it mid-window rescales a partial sum — silently wrong
  (regression-tested in `src/optim_tests.zig`).
- **Checkpoints only at window boundaries.** Accumulated gradients live only
  in the GradStates and are NEVER serialized (`optimizer.fucina` stores
  moments, not grads) — a save or crash mid-window drops the partial sum on
  resume. Note the two counters: the trainers' `step_counter`
  (`trainer_state.json` `.step`) counts MICRO-batches (dropout seed streams,
  one per `loss`/`lossExt` call), while the optimizer slots' `step` counts
  MACRO steps; resume must land on a boundary, i.e. `step % accum_steps == 0`
  (the optional `accum_steps` field in `training_checkpoint.TrainerState`
  records the window size).
- **LR schedules are keyed by MACRO step** (one `sched.apply` per optimizer
  step, not per micro-batch).
- **Determinism contract.** Same data + seed + thread config ⇒ bitwise-
  reproducible accumulated grads and post-step params across runs: sequential
  micro-batch backwards are ordered by construction, given the repo's
  per-backward determinism (§8 header note in `src/optim.zig`: a tensor fed
  by 3+ async heavy branches inside ONE backward accumulates in completion
  order; two contributions are order-safe). It is NOT bitwise-equal to a
  hypothetical single N× batch — fp summation order differs (equal-size
  micro-batches match a true big batch to ~1e-5 relative, asserted in
  `src/optim_tests.zig`).

## 5. Loss: cross-entropy options

`x.crossEntropy(ctx, .class, labels)` is plain mean CE.
`x.crossEntropyExt(ctx, .class, labels, options)` adds the PyTorch-parity
knobs (`exec.CrossEntropyOptions`, comptime — `src/exec/loss.zig:33`):

- `ignore_index` — sentinel label: those positions contribute zero loss and
  zero gradient and are excluded from the `.mean` denominator (PyTorch
  semantics). Documented divergence: when EVERY position is ignored the loss
  is 0 (and gradients are zero), not PyTorch's NaN. Labels must be
  `< class_count` or equal the sentinel — an out-of-range sentinel like
  `maxInt(usize)` works.
- `reduction = .mean | .sum | .none` — `.none` returns per-position losses
  with the class tag removed (ignored positions get 0); the VJP handles both
  scalar and per-position upstream gradients.
- `label_smoothing` in [0, 1) — target = (1-eps)·onehot + (eps/K)·uniform
  over all K classes (PyTorch semantics, target class included).

The row kernels (softmax fwd/bwd, CE fwd/bwd) are SIMD (ggml-style `vexpf`
range reduction in `src/backend/vector/primitives.zig`) and parallel over
rows. Determinism: each row writes its loss to a per-row buffer and the
`.mean`/`.sum` reduction is ONE serial sum in row order (`src/exec/loss.zig:73-74`),
so the loss is bitwise stable across thread counts. Measured (M1 Max,
ReleaseFast, `zig build bench-ce`): CE forward at 1024x151936 354.5 ms →
~18 ms, backward 426 ms → ~22 ms (~19-20x); softmax 13-21x across shapes.
The autograd CE node saves the forward's per-row {max, sum_exp} (8 B/row),
so its backward emits final gradients in ONE pass — bitwise identical to
recompute, 22.7 → 14.0 ms at 1024x151936.

**Fused projection + loss:** `x.linearCrossEntropyExt(ctx, &w, labels, options)`
computes `crossEntropyExt(x·wᵀ)` as ONE differentiable op — `x` is
`[row, shared]`, `w` is `[class, shared]` (rank-2, shared tag last, f32),
gradients flow to BOTH operands, and the same reduction contract applies.
The logits never enter the graph: the record saves them with the row stats,
and the backward overwrites them IN PLACE with the logit gradient before the
two gradient GEMMs, so the full `[rows, classes]` gradient never costs a
second buffer (−622 MB peak and ~4% faster than the composed dot +
crossEntropyExt backward at 1024x151936x1024 on M1). The record is
single-use: re-running its VJP errors with
`LinearCrossEntropyBackwardConsumed` instead of computing garbage (a plain
repeated `backward()` on the same graph is already rejected upstream with
`error.BackwardAlreadyRun`) — rebuild the forward to backward again
(accumulation loops already do).

Normalization for non-RMSNorm architectures: `layerNorm(ctx, tag, eps, .{})` /
`layerNorm(ctx, tag, eps, .{ .weight = &w, .bias = &b })` (torch.nn.LayerNorm
semantics — biased variance; affine = fused normalize·w+b) plus `max` / `min` /
`variance(ddof 0|1)` reductions, all with full VJPs (per-operand pruning;
dweight/dbias bitwise identical for any thread count), so GPT-2/BERT-class
models are now expressible. Cost ~1.5-2x rmsNormMul at LLM shapes (`bench-ce`
rows alongside rmsNormMul).

## 6. Dropout and the RNG discipline

`x.dropout(ctx, p, seed)` is inverted dropout: element i keeps `x[i]/(1-p)`
iff the 53-bit uniform of `rng.at(seed, i)` is below `1-p`, else 0. The mask
is NEVER stored — forward, backward, and any checkpoint recompute regenerate
it from (seed, element index), so the op is a pure function of (input, p,
seed), bitwise identical for any thread count (counter-based access makes
every element independently computable). `p == 0` is a zero-copy identity
view. Eval mode is caller-side: don't call dropout at eval.

`src/rng.zig` is the repo-owned deterministic RNG: `splitmix64` (sequential),
`at(seed, i)` (counter-based O(1) random access into the same stream),
`uniformFill`, `gaussianFill` (Box-Muller), `kaimingUniformFill` (PyTorch
`kaiming_uniform_` with a=sqrt(5) — the `nn.Linear` / LoRA-A default init),
`normalFill`. Seed discipline: derive a fresh seed per call site, e.g.
`rng.at(base_seed, step * n_layers + layer)` — reusing a seed reuses the
mask. The (seed → values) mapping is a CHECKPOINT CONTRACT (the same
rationale as APOLLO's regenerated projections): deliberately not
`std.Random`, so it survives Zig toolchain upgrades.

## 7. Gradient (activation) checkpointing

`fucina.checkpoint(ctx, block, inputs)` (`src/ag/checkpoint.zig`) trades
compute for memory: the forward runs `block` GRAD-FREE under an inner exec
scope — every intermediate is released the moment the block returns — keeping
only refcounted views of the inputs plus one deep copy of the output. When
backward reaches the checkpoint it re-runs the block to rebuild the subgraph,
backprops the incoming gradient through it, and hands the input gradients to
the outer engine. O(inputs + output) retained instead of O(intermediates): an
8-block chain retains 8 scope entries vs 24 plain, with bitwise gradient
parity (both asserted in `src/ag/checkpoint_tests.zig`).

- `block` is a comptime function `fn (*ExecContext, *const Tensor(..), ...)
  !Tensor(..)` over f32 facade tensors; `inputs` is a tuple of pointers
  matching its parameters. The block always runs under a scope, so the
  defer-deinit forward idiom works unchanged inside it.
- `block` must be deterministic and pure in its inputs: the recompute must
  rebuild the exact forward values. Dropout under a checkpoint replays
  bitwise BY CONSTRUCTION (the mask is a function of the stored seed, §6) —
  provided the seed travels through stored state, not ambient RNG.
- `checkpointWithContext(ctx, block, extra, inputs)` carries everything that
  is NOT a differentiable f32 input — frozen weights (quantized/f16/bf16
  constants), RoPE tables, config values, layer struct pointers — as `extra`,
  stored BY VALUE in the backward node. Contract: plain data/pointers that
  remain valid until backward completes (no deep copy, no refcount);
  everything reachable through `extra` is a constant (never receives
  gradients); the block stays deterministic in (`extra`, `inputs`).
- Recomputes are serialized by a module-level mutex (re-run facade ops adopt
  into ctx exec-scope state, which is not thread-safe); a nested checkpoint inside
  a block is rejected with `error.NestedCheckpointRecompute` instead of
  deadlocking.

Real-model cost: per-layer checkpointing of the Qwen3-0.6B LoRA fine-tune
(`--checkpoint-layers`, §9) reproduced digit-identical losses at ~+8.5% step
time.

## 8. Checkpoint files

Canonical resumable checkpoints are directories. The portable tensor artifact
is a clean safetensors file; Fucina-only resume state lives beside it:

```text
checkpoint/
  model.safetensors        # or adapters.safetensors
  optimizer.fucina         # native optimizer frames
  trainer_state.json       # written last; commit sentinel
```

- Parameter values, two formats:
  - `optim.saveTensors(writer, .{ &w1, &b1, ... })` / `loadTensors` —
    legacy positional FZT1, f32-only: the loading program must list the same
    tensors in the same order (shapes validated).
  - `optim.saveStateDict(allocator, writer, entries)` / `loadStateDict` —
    named safetensors state dict: per entry a unique name, dtype
    (f32/f16/bf16, raw byte passthrough), shape, and data
    (`NamedTensor` / `NamedTensorMut` entries). Load reads one safetensors
    prefix and matches stream entries BY NAME, so entry order is free; strict
    (the default) demands a one-to-one match, non-strict skips unknown stream
    entries. A standalone state dict is a valid safetensors file.
- `opt.saveState(writer)` / `opt.loadState(reader)` — moments, step counts,
  and the structural config fields, validated on load
  (`error.CheckpointConfigMismatch` on e.g. a changed APOLLO rank or scale
  type; per-slot dims are validated too). Formats are v3
  (FZAD/FZA3/FZM3/FZP3/FZS3; FZO3 for OptimizerSet) and v4; v2 readers are
  dropped. When every state buffer is f32 (the default) the v3 frames are
  written BYTE-IDENTICALLY to pre-`StateDType` builds, so older builds keep
  reading new f32 checkpoints. When any state buffer is non-f32 the v4 magics
  are written instead (FZD4/FZA4/FZM4/FZS4 for Adam/AdamW/Muon/SGD; Apollo
  state stays f32, always FZP3): same layout except every state buffer is
  prefixed by one u8 `StateDType` tag (0 = f32, 1 = bf16). Loaders accept
  both versions and require the stored dtype to match the configured one
  EXACTLY — v3 implies f32 everywhere, so loading a v3 file into a
  bf16-configured optimizer (or any dtype mismatch under v4) errors with
  `error.CheckpointDtypeMismatch`. There is deliberately NO implicit
  f32<->bf16 conversion on load: it would silently break the bit-exact-resume
  contract below. Slots are matched BY NAME — explicit via `addParamNamed` /
  `addFallbackParamNamed`, otherwise the auto-name "param<i>" from the slot's
  index — so named params may be re-registered in ANY order within their slot
  list (permuted-registration resume stays bit-exact, tested for
  AdamW/Muon/Apollo plus the bf16 v4 variants); unnamed params must keep
  their relative registration order to reproduce their auto-names.

Contracts to respect:

- **A failed load leaves the target partially restored.** Treat load errors as
  fatal for that model/optimizer instance.
- **Bit-exact resume** holds when the surrounding forward/backward replay is
  deterministic. Optimizer state restore is always byte-exact; elementwise
  updates are parallelized as fixed-boundary, element-independent maps, so
  they are bitwise identical for any thread count. The one caveat: a tensor
  receiving 3+ gradient contributions whose consumers' heavy backward
  branches run async (matmul, attention, causal conv1d — anything past the
  256 MiB work threshold in `src/ag/core.zig`) is accumulated in completion
  order. Two contributions are always order-safe (IEEE addition commutes).
- **APOLLO projections are regenerated, not stored**: P is a deterministic
  function of (seed, step / update_proj_gap) computed by the repo-owned
  splitmix64 + Box-Muller generator (`src/rng.zig`, §6) — deliberately not
  `std.Random`, so the mapping survives Zig toolchain upgrades.

`zig build spirals` proves the full composition: every optimizer (and the
groups+schedule+clip combo) checkpoints at the halfway step, resumes into
fresh objects, and asserts bit-identical final parameters.

## 9. LoRA + LLM fine-tuning

**Adapters** (`fucina.lora`, `src/lora.zig`). `lora.Adapter(in_tag, out_tag)`
learns `y = base(x) + (alpha/r) * dropout(x)·A^T·B^T` over a FROZEN linear —
anything `dot` accepts as a frozen RHS (f32/f16/bf16/block-quantized
constants). A: [r, in] kaiming-uniform (seeded, deterministic), B: [out, r]
zeros, so the initial delta is exactly zero. The rank axis uses the reserved
tag `.lora_r`.

- `init(ctx, in_dim, out_dim, .{ .rank, .alpha, .dropout_p }, seed)` —
  effective scaling is alpha/rank. A/B are caller-owned variables (never
  scope-adopted).
- `delta(ctx, x, dropout_seed)` / `apply(ctx, x, base, dropout_seed)` — a
  null `dropout_seed` is eval (skips dropout). Composite-op contract: calls
  whose result will be `backward()`'d MUST run under an exec scope (interior
  tensors release on return; the scope keeps the graph alive).
- `registerParams(opt, "prefix")` → `addParamNamed` as
  "prefix.lora_a"/"prefix.lora_b"; `namedTensors` / `namedTensorsMut` for
  safetensors persistence.
- `mergeInto(ctx, &w)` folds the adapter into an f32 base IN PLACE (the
  facade's `data()` gate restricts this to no-grad tensors — exactly the
  frozen-base case); `mergeF16` widens/merges/casts back and returns a new
  f16 tensor. Quantized bases are still NOT mergeable in place — deliberately,
  even though the f32 → K-quant encoders exist (`quantizeRowForDType` in
  `src/backend/quant.zig` / `gguf.encodeF32`): dequantize→merge→re-encode
  compounds quantization error. For in-memory use, dequantize to f32
  (`.to(ctx, .f32)`) and merge into the copy; for files, merge into a dense
  f32/f16 base and quantize the RESULT (the export loop below).

**Qwen3 fine-tuning** (`llm.qwen3.train`, `src/llm/qwen3/train.zig`).
`Trainer(targets)` is a full-sequence grad-capable forward over a frozen GGUF
`qwen3.Model` — op-for-op the inference math (norms, fused q/k-norm+RoPE,
grouped causal attention, SwiGLU) with LoRA deltas on the projections
selected at comptime by `targets` (q/k/v/o/gate/up/down). Frozen projections
route through the differentiable const-RHS `dot` on each weight's plain value
tensor — the packed `linearSeq` fast paths stay inference-only — so gradients
flow to the f32 activations only and weight memory stays quantized/f16. The
base model is never written; the only parameters are the adapters' A/B. MoE
configs are rejected (`Error.MoeUnsupported`) in v1.

- `loss(ctx, tokens, labels)` — mean CE with `qwen3.train.ignore_index`
  (`maxInt(usize)`) masking prompt positions; requires an open exec scope
  (`Error.ExecScopeRequired` otherwise). Each call advances the step counter
  that derives per-(step, layer, projection) dropout seeds via `rng.at` —
  replayed bitwise under checkpointing.
- `trainer.checkpoint_layers = true` — one `checkpointWithContext` per layer,
  frozen layer state traveling in `extra`; bitwise-equal losses and adapter
  grads (tested).
- `registerAllParams(opt)`; `saveAdapters`/`loadAdapters` (safetensors, names
  "layers.<i>.<target>.lora_{a,b}"; `loadAdaptersWithOptions` threads
  `state_dict.LoadOptions` — e.g. name `aliases` — while `loadAdapters` stays
  the strict arm); `evalLastLogits` for generation.
  `saveAdapters` writes only the portable adapter tensors; resumable fine-tune
  runs store the dropout step counter, seed, LoRA config, and LR in
  `trainer_state.json`.

**The walkthrough** (`examples/finetune.zig`):

```sh
zig build finetune -Doptimize=ReleaseFast -- --steps 30
```

Model GGUFs are not part of the repo — the default model is
`models/Qwen3-0.6B-Q4_K_S.gguf`; see `RUNNING-MODELS.md` for where to
download each model (or pass your own with `--model`).

LoRA on q+v (rank 8, alpha 16, lr 1e-3, AdamW — the defaults) over
Qwen3-0.6B-Q4_K_S on a tiny built-in SFT set with prompt masking: ~932
ms/step, 38.2 tok/s supervised throughput (M1 Max); loss 5.77 → 2e-4 in 30
steps; prints a BEFORE/AFTER greedy generation and saves a checkpoint
directory with clean `adapters.safetensors`, native `optimizer.fucina`, and
`trainer_state.json`. `--save-every N` writes PyTorch-style periodic
subdirectories under the save directory (`checkpoint-step-N`).

**Data comes from `llm.data` (`src/llm/data.zig`).** The SFT
plumbing finetune used to hand-roll is a reusable utility: `SftText`
(JSONL via `fromJsonl` with configurable keys, or zero-copy `fromPairs`),
`encodePair`/`encodePrompt` (chat-template render + `encodeRaw` + next-token
shift with `ignore_index` prompt masking; the response and stop marker are
encoded separately from the prompt — encoding a concatenation would change
token IDs), and a deterministic `Loader` (`.sequential` or `.shuffled`;
`State{seed, epoch, index}` persists in `trainer_state.json` as
`data_seed/data_epoch/data_index`, so resume CONTINUES the stream). The
`(seed, epoch) → permutation` mapping — Fisher–Yates over a `splitmix64`
stream seeded by `rng.at(seed, epoch)` — is a CHECKPOINT CONTRACT
(golden-pinned in `data_tests.zig`, same rationale as §6's RNG). finetune
flags: `--data PATH.jsonl`, `--shuffle`, `--data-seed N` (defaults to
`--seed`; order flags are not persisted — pass the same ones on resume). A
padded batcher is deliberately absent: the trainers take one flat sequence,
and gradient accumulation (§4) is the batch mechanism on this runtime.

**Gradient verification — how we know the gradients are right.** Three
independent angles:

- **Full-stack finite differences** (`src/llm/qwen3/train_tests.zig`).
  Central differences (eps 5e-3) through the ENTIRE `Trainer.loss` — CE
  masking inside the differentiated function, seq 64 so the query-tiled
  attention forward is on the path, a 2-step warmup so B != 0 and the
  A-grads are live — over 280 adapter elements (every element of one
  adapter, samples of all the rest) vs the analytical backward: cosine
  similarity 0.999998, worst |dev| 1.9e-4 against tol ≥ 1e-3 (f32-FD noise,
  ≥5x margin). All-masked batch → loss 0 and exactly-zero adapter grads;
  frozen base weights carry no GradState; checkpointed grads are
  bitwise-equal to plain, which transfers the FD verdict to the
  checkpointed path without a second FD loop.
- **PyTorch golden parity** (`src/llm/qwen3/train_golden_tests.zig`,
  generated by `tools/gen_qwen3_train_goldens.py`, torch
  2.12.0, seed 0xF0C1AA, sha256-auditable data section). An INDEPENDENT
  torch replica of `Trainer(.{})` (NeoX rope half mode, fused qk-norm
  order, eps-inside-sqrt rmsnorm, up·silu(gate), GQA, CE mean-over-valid)
  bakes weights + expected loss/adapter-grads into Zig constants. Parity at
  seq 64 (tiled attention path): loss 1.7e-7 relative native / 8.5e-8
  scalar; worst grad element 2.7e-7 — >100x headroom inside the test
  tolerances on both backends. No architecture mismatch found.
- **Real-model evidence** (`zig build finetune -Doptimize=ReleaseFast --
  --verify-grads`, replaces the training run). Five causal checks through
  the production path (quantized frozen weights, tiled attention, fused
  kernels) on Qwen3-0.6B: zero-structure at init (B == 0 ⇒ dL/dA == 0
  identically while dL/dB != 0); grad-norm audit — 112 adapter-grad norms
  (28 layers × {q,v} × {A,B}): min 0.150 / median 0.597 / max 2.397, no
  zeros, no non-finite; first-order Taylor test with a random-direction
  noise floor — through Q8_K-quantized ACTIVATIONS the loss is locally a
  staircase, so R = (L0−L1)/(lr·‖g‖²) is adjudicated only where the
  predicted drop clears 3x the measured floor: when first measured
  (2026-06-11), R = 0.89 at lr 3e-3 on the Q4_K_S base (in [0.7, 1.1];
  curvature-consistent with 0.96 at the noise-dominated 1e-3), and on the
  f16 base the floor collapses ~10-30x giving R = 0.86/0.94/0.80 at
  signal/noise 5-88 — the cross-run pins the small-lr anomaly on the
  activation-quantization staircase, NOT the gradients. Known open issue:
  at the published commit the same Q4_K_S Taylor leg reads R ≈ 0.59 at lr
  3e-3 and `--verify-grads` exits non-zero; the torch-golden and FD checks
  above still pass, so the discrepancy is confined to this
  staircase-sensitive probe and has not been root-caused yet; frozen
  ablation — repeated loss bitwise identical, base
  weights provably grad-free (runtime `requiresGrad` plus a comptime
  assert: quantized weight types carry no `grad_state` field at all);
  held-out generalization — CE on a never-trained pair −62.5% vs raw init
  and generation flips to the trained style. ReleaseSafe reproduces the
  ReleaseFast report bitwise.

**The complete loop: fine-tune → merge → quantize → serve anywhere.**
`zig build export-gguf` (`tools/export_gguf.zig`) closes training back into
GGUF. Proven end-to-end: the merged model answers in the fine-tuned style
under llama-cli, and the f16→q4_k export loads in Fucina with top-1 logit
parity AND in llama-bench as "Q4_K - Small".

```sh
# 1. fine-tune: checkpoint directory with adapters.safetensors + optimizer.fucina
#    (download the f16 GGUF first — see RUNNING-MODELS.md)
zig build finetune -Doptimize=ReleaseFast -- \
    --model models/Qwen3-0.6B-f16.gguf --steps 30 --save /tmp/qwen3-lora
# 2. merge the adapters into the dense f32/f16 base (--alpha = training-time alpha; finetune default 16)
zig build export-gguf -Doptimize=ReleaseFast -- \
    --from-gguf models/Qwen3-0.6B-f16.gguf --adapters /tmp/qwen3-lora \
    --alpha 16 --out /tmp/qwen3-tuned-f16.gguf
# 3. quantize the merged model (byte-exact ggml-parity encoders; also f16/bf16/f32/q8_0/q5_k/q6_k)
zig build export-gguf -Doptimize=ReleaseFast -- \
    --from-gguf /tmp/qwen3-tuned-f16.gguf --dtype q4_k --out /tmp/qwen3-tuned-q4_k.gguf
# 4. serve in Fucina ...
zig build qwen3 -Doptimize=ReleaseFast -- /tmp/qwen3-tuned-q4_k.gguf --chat "What is the capital of France?"
# ... or in llama.cpp (or any GGUF consumer)
llama-cli -m /tmp/qwen3-tuned-q4_k.gguf
```

Contracts: merge and `--dtype` are separate passes BY DESIGN (one combined
pass would chain-requantize); merge needs a dense f32/f16 base
(`mergeInto`/`mergeF16` underneath — a quantized base errors; quantize AFTER
merging, as above). Transcode policy is llama-quantize's: matrix weights
only (n_dims ≥ 2, name `*.weight`, not "norm"; norms/1D keep their stored
type), K-quant targets fall back to the source type when the innermost dim
is not %256, and already-quantized sources error instead of
chain-requantizing (re-emit those verbatim). Adapters are matched by the
`Trainer` names (`layers.<i>.<target>.lora_{a,b}` → `blk.<i>.*.weight`); the
adapter checkpoint stores A/B but not alpha — pass the training-time value.
The writer side is exact: a verbatim re-emit of the 449 MiB Q4_K_S GGUF is
byte-identical to the original (`cmp` clean).

## 10. bf16 + mixed precision: what exists, what was deferred

**Exists.** Frozen-weight mixed precision is fully differentiable: `dot`
against a CONSTANT f16/bf16/quantized RHS routes gradients to the f32 LHS
only (`ConstRhsDotBackward`), and attention reads f16 KV. (Opt-in
`--cache-type q8_0` halves cache memory again — 59.5 vs 112 KiB/token on
0.6B — but it is the CAPACITY option, not a speed one: decode attention is
compute-bound on M1 and the dequant costs ~2.3x the attention phase at 2048
ctx, so f16 stays the default; the differentiable q8_0 facade entry dequants
the cache once, cache constant, q-grad only.) The bf16 leg: a mixed f32 x
bf16 TransB kernel (`src/backend/vector/gemm.zig`) widens the bf16
weights in-register (u16 << 16, exact) and accumulates everything in f32 —
unlike the f16 twin, which casts the LHS down and accumulates in half
precision. bf16 GGUF weights stay RESIDENT at 2 B/weight
(`weights.zig` `.bf16` arm: raw-bits load, `linearSeq` through the mixed
kernel, typed `getRowsAs`, `fuseLinear`) instead of widening to f32 at load;
bitwise parity vs the widened reference is unit-tested, and a frozen bf16
base is differentiable via `ConstRhsDotBackward(.bf16)` — the only
DIFFERENTIABLE bf16 route. Model-level (real Fucina-exported 0.6B bf16 GGUF,
M1 Max): same top-5 ranking as f16 within bf16 rounding; decode 54.7 vs
f16's 65.5 tok/s; pp32 1.52x f16's time. The kernel-level prefill gap was
already measured at ~2.06x the f16 twin — M1 has no bf16 FMA; an ISA gap,
not an implementation one.

**bf16 optimizer STATE (2026-07-03).** Optimizer
moment/momentum buffers can be stored in bf16 (`optim.StateDType`, §3;
checkpoint v4 frames, §8). This is ORTHOGONAL to every deferred item below:
step math, params, activations, and gradients all stay f32 — only the
between-steps storage of m/momentum (and, opt-in, AdamW/Adam v) narrows.

**Deferred deliberately.**

- f16/bf16 activations + gradients: needs a dtype-generic
  FloatTensor/GradState and a duplicated VJP surface — a ground-up rebuild
  of the autograd layer (a full comptime-dtype-generic runtime migration,
  evaluated and deliberately not undertaken), not an increment.
- Loss scaling: pointless while gradients are f32 (its only job is rescuing
  f16 gradient underflow).
- f32 master weights: pointless while the trainable params are already f32.

**Revisit when**: an x86 AVX512-BF16/AMX target makes bf16 GEMM the actual
training win, or LoRA-scale profiling shows activation bandwidth dominating
step time.

## 11. Performance

**GPU GEMM offload (`-Dgpu=metal`, 2026-06-12):** big f32 GEMMs — exactly the
training forward/VJP shapes — can run on the Apple GPU (default gate: m·n·k ≥
2^30 ≈ 1024³, the measured M1 Max crossover vs Accelerate; the 2048×1024×1024
train shape is +83%). Quantized-frozen LoRA fine-tuning at seq≤256 has no
qualifying f32 GEMMs and is measurably unchanged (never-a-loss). GPU
accumulation order differs from CPU, so loss curves diverge in the last ulps
versus a CPU run. The build option and its environment knobs are described in
`AGENTS.md`.

`zig build bench-optim -Doptimize=ReleaseFast` measures step kernels at
Qwen3-0.6B-class shapes. M1 Max snapshot (2026-06-10, native backend +
Accelerate, 50 timed iters, range over 3 runs; one transformer block = 15.7M
params). The elementwise rows wobble with the machine's thermal/scheduler
state (see `BENCHMARK.md`); the GEMM-bound rows are stable:

| optimizer | ms/step (block) | eff. GB/s | x28 layers | embedding 155.6M |
| --- | --- | --- | --- | --- |
| sgd | 1.8–3.0 | 63–105 | 50–84 ms | 14–20 ms |
| sgd-momentum | 2.5–4.8 | 66–125 | 70–135 ms | 19–21 ms |
| adamw | 5.2–6.0 | 74–85 | 145–167 ms | 42–52 ms |
| muon | 446–457 | — | ~12.6 s | (fallback = adamw) |
| apollo-r256 | 44–46 | — | ~1.27 s | (fallback = adamw) |
| apollo-mini | 23–24 | — | ~0.66 s | (fallback = adamw) |

Reading the table:

- The elementwise optimizers run near memory bandwidth (fused single-pass
  updates, chunked over the worker pool). A full Qwen3-0.6B AdamW step costs
  ~200 ms of optimizer time — small next to the forward/backward.
- **Muon's cost is Newton-Schulz GEMM FLOPs** (~395 GFLOP per block in f32 at
  5 iterations), not loop overhead. That is the algorithm; the reference
  mitigates it with bf16 on GPU. Future work if it matters on CPU: SYRK for
  the Gram matrix (halves one GEMM) and/or fewer NS steps.
- APOLLO sits between: one projection GEMM per matrix plus compressed-space
  moments; Mini (rank 1) is cheapest.
- Always train in `ReleaseFast`; validate in `Debug` (the DebugAllocator
  catches lifetime mistakes — the spirals example panics on leaks).

**bf16 state rows (2026-07-03, §3).** `bench-optim` grew
`sgd-momentum-bf16` / `adamw-bf16-m` / `adamw-bf16-mv` / `muon-bf16` rows.
Quiet-machine M1 Max numbers (ReleaseFast, 50 timed iters): block adamw
3.605 → 2.028 ms (m bf16, 1.78x) / 2.014 ms (m+v bf16, 1.79x); sgd-momentum
2.284 → 1.450 ms (1.58x); muon ≈ parity (432.9 vs 445.1 ms — Newton-Schulz
GEMMs dominate); embedding [151936, 1024] adamw 34.28 → 18.91 ms (m bf16,
1.81x) / 17.05 ms (m+v bf16, 2.01x). The step is bandwidth-bound, so
narrower state is FASTER, not slower. Two implementation facts behind the
speedup: LLVM does NOT auto-vectorize these fused sqrt/div update loops on
aarch64 (the f32 arms run scalar and hide behind memory bandwidth at 8
threads), so the bf16 arms are HAND-vectorized (`state_vec_len` in
`src/optim.zig`); and the vector f32→bf16 narrow keeps the scalar helper's
NaN-quieting guard and is proven lane-exact against it by an exhaustive 2^32
sweep during development plus the exact-parity unit tests (vector body AND
scalar tail).

Companion training benches: `bench-ce` (softmax/CE/layerNorm row kernels,
numbers in §5) and `bench-scatter` (the embedding-gradient scatter-add — the
destination-row-partitioned parallel kernel is 2.5-4.8x over serial at
151936x1024 and bitwise-identical for any thread count).

Attention forward at `q_seq >= 48` (d ≤ 256) now dispatches to the
query-tiled online-softmax kernel (`BENCHMARK.md` 2026-06-11): values agree
with the 3-pass per-query kernels to ~1e-6 relative, not bitwise; below the
threshold and at decode nothing changes. Gradient correctness is unaffected —
the attention backward recomputes scores/softmax internally from the saved
q/k/v, never from the forward output.

## 12. Pitfalls checklist

- Deinit an intermediate before `backward()` with NO scope open → dangling
  graph node (UB). Use an exec scope; under one, deinit on op results is a
  safe no-op and the graph survives to backward.
- Use a scope-owned tensor after `closeExecScope` → use-after-free (the one
  borrow hazard the `scope_owned` flag cannot remove).
- Forgot `zeroGrad()` → gradients accumulate across steps (sometimes wanted:
  that *is* gradient accumulation; divide the loss accordingly — the full
  recipe, normalization arms, and clip/checkpoint/schedule rules are in §4
  "Gradient accumulation").
- Reading `param.data()` on a variable → `error.MutableDataRequiresNoGrad`;
  use `dataConst()`, or mutate through the optimizer.
- Muon/Apollo on an embedding or head via `addParam` → silently wrong
  algorithm for those weights; route them with `addFallbackParam`.
- Changed optimizer config between save and load → caught for structural
  fields (`CheckpointConfigMismatch`), NOT for lr (schedules legitimately
  change it — re-attach and re-apply instead).
- Registering one tensor twice with the same optimizer →
  `error.DuplicateParam`. Registering it with two *different* optimizers
  (e.g. two groups of an OptimizerSet) is NOT detected: it double-steps and
  double-counts in `clipGradNorm`. Keep group memberships disjoint.
- `nesterov` SGD with dampening or zero momentum → panics at init in every
  build mode (PyTorch constructor rule).
- Reusing a dropout seed across calls → identical masks (correlated dropout).
  Derive a fresh seed per step/layer with `rng.at` (§6).
- A checkpoint block that is not deterministic in (`extra`, `inputs`) →
  silently wrong gradients (the recompute diverges from the forward). RNG
  must come from stored seeds; `extra` pointees must outlive backward.
- A nested `checkpoint` inside a block → `error.NestedCheckpointRecompute`
  (caught; the recompute lock is not reentrant).
- LoRA `delta`/`apply` (or any composite op) backward()'d without a scope →
  the same dangling-node UB as the first bullet; `qwen3.train.Trainer.loss`
  checks and returns `Error.ExecScopeRequired` instead.

## 13. Evolution strategies (gradient-free training)

`fucina.es` (`src/es.zig`) trains WITHOUT gradients: it is a faithful
reimplementation of the ES-at-scale algorithm ("Evolution Strategies at
Scale: LLM Fine-Tuning Beyond Reinforcement Learning", arXiv:2509.24372;
reference `github.com/VsonicV/es-at-scale`) — deliberately vanilla OpenAI-ES
with the reference's simplifications kept intact:

```text
eps_n ~ N(0, I)                               n = 1..population
R_n   = reward(theta + sigma * eps_n)         (forward passes only)
C_n   = (R_n - mean(R)) / (std(R) + 1e-8)     (z-score, f64 stats, ddof = 0)
theta += (alpha / population) * sum_n C_n * eps_n
```

No antithetic pairs, no rank shaping, no optimizer state, and no 1/sigma in
the update (folded into alpha; the reference default is `alpha = sigma/2`,
`sigma = 0.001`, `population = 30` — also the `es.Config` defaults). Because
the signal is a scalar reward per population member, ES composes with
anything you can score from a forward pass — generation-based rule rewards
included — and every parameter is fair game, gradients or not: `es.Trainer`
registers f32 variables, f32 constants, and typed f16/bf16 tensors alike
(`addParam`/`addParamNamed`), or an entire `ParamRegistry` in one call
(`addRegistry`, frozen entries included; tied storage deduplicates like
torch `named_parameters()`).

```zig
var trainer = try fucina.es.Trainer.init(allocator, .{ .sigma = 0.01, .population = 16 });
defer trainer.deinit();
try trainer.addParam(&w);

const Evaluator = struct {
    w: *const W,
    pub fn eval(self: *const @This(), member: usize) !f32 {
        _ = member; // the perturbation is already applied in place
        return score(self.w); // any forward-pass metric
    }
};
for (0..iterations) |_| {
    const stats = try trainer.step(&ctx, &Evaluator{ .w = &w });
    _ = stats; // pre-normalization mean/std/min/max reward
}
```

**Scale mechanics.** Noise is never stored: a member's perturbation is a pure
function of (seed, iteration, member, slot, element) through the
counter-based gaussian, so `perturb`, `restore`, and `update` regenerate it
on the fly — O(1) memory beyond the parameters, the reference's headline
trick. Regeneration is the dominant cost of full-parameter ES on CPU
(~24 full-parameter passes per iteration at population 8), so es.zig draws
through `rng.gaussianFillAtFast`: the same splitmix64 pair stream and
uniforms as the scalar kernel with vectorized ~1-ulp polynomial
transcendentals, several times faster per value (a distinct
(seed -> values) checkpoint contract from `gaussianFillAt`; APOLLO/dropout
stay on the scalar mapping). Because any element is O(1)-addressable, the
kernels chunk across the worker team with results bitwise identical to the
serial loop for any thread count (optim.zig's parallelMap discipline).
`antithetic = true` (opt-in, Salimans-2017 mirrored sampling — the other
component the reference stripped) pairs members as (+eps, -eps) sharing one
draw: the update folds each pair to `(C+ - C-) * eps` (half the update-side
regeneration) and mirrored sampling's variance reduction classically buys
roughly another 2x progress per iteration; requires an even population and
is part of the noise contract (persisted as `es_antithetic`). Two
evaluation shapes:

- **In place** (big models): `perturb`/`restore` mutate the registered
  buffers; one model serves the whole population, each forward saturating
  the cores. `step` is the sequential driver. `restore_mode = .regenerate`
  (default, reference semantics: subtract the regenerated noise, exact up to
  `(x+t)-t` float drift) or `.snapshot` (memcpy-back, bitwise, costs one
  parameter copy).
- **Member-parallel replicas** (small parameter sets: adapters, MLPs):
  `materializeMember` writes `theta + sigma*eps_n` into caller-owned replica
  buffers without touching shared theta, and `evaluateMembers` fans members
  out over OS threads — ES's embarrassing parallelism on CPU.

On `-Dgpu=cuda` builds, slots whose storage is GPU-resident (dense f16
model weights load that way) run perturb/restore, the population update,
and anchored weight decay as device kernels — the CUDA port of the noise
contract is bitwise identical to the CPU kernels for any launch geometry
(round-to-nearest intrinsics, no FMA contraction), so checkpoints are
device-independent; non-resident slots, bf16, and active stream caches
fall back to the CPU path per slot. On unified-memory Metal the CPU
kernels already mutate shared pages the GPU reads zero-copy, so no device
arm exists there.

**Noise schemes.** `.iid` (default): every (member, slot) gets an
independent stream. `.correlated`: a member reuses ONE stream across slots,
so same-length slots receive identical noise — this mirrors the reference
library, which reseeds the same generator per tensor (an acknowledged
artifact its authors kept; the paper's results used it). Both are checkpoint
contracts: `(config.seed, iteration)` fully regenerate the population, so a
resume needs only the iteration counter (`TrainerState.es_*` fields — there
is no optimizer.fucina in an ES checkpoint).

**Parity evidence.** Three layers, mirroring the optimizer goldens:
`tools/gen_es_goldens.py` replicates the repo RNG bit-level and the update
algebra in numpy → tolerance goldens in `src/es_tests.zig` (tolerances
because libm `log`/`cos` vary by an ulp across platforms); a test-local
straight-line serial reference pins the chunk-parallel path bitwise; and
`tools/check_es_parity.py` runs the ACTUAL reference code (perturb / restore
/ z-score / update from a local `refs/es-at-scale` clone, stub model_runner,
no vLLM needed) against a torch transcription of es.zig's algebra on
identical noise — bitwise `torch.equal` on f32/f16/bf16, both schemes. The
one deliberate substitution vs the reference is the RNG itself (repo-owned
splitmix64 instead of torch Philox — a checkpoint contract here); note the
reference repo is Academic-Public-License, so it is compared against, never
ported into the tree. AWD gets the same treatment: the checker also runs the
actual `apply_reference_weight_decay_`/`update_parameters_from_seeds` from a
local `refs/es-awd` clone (github.com/kschweig/es-awd, MIT-adjacent LICENCE
in-repo) against a transcription of es.zig's anchor kernel — bitwise on
f32/f16/bf16 for l1 and l2, decay and full update-then-decay sequence. One
documented cross-reference nuance: es-awd's own ES update uses torch's
fused `add_(alpha=)` and differs from es-at-scale's explicit mul-then-add
by one f32 ulp; es.zig follows es-at-scale.

**Reward design and shaping (the stability lever).** The reference's
z-score is affine, so it PRESERVES outlier magnitude: with an unbounded
reward like raw -CE, one catastrophically-perturbed member can carry several times
the coefficient of every well-behaved member, degenerating the update into
"flee the bad direction" (the signature is descent, then growing population
std and collapse). That is the exact "outlier individuals" pathology rank
shaping was introduced to remove (Salimans et al. 2017); plain z-score is
sufficient only when task rewards are bounded by construction, as the
reference's are. Two remedies, composable:

- **Bounded rewards.** Prefer rewards in a fixed range; a member that breaks
  catastrophically saturates the floor instead of dominating the z-score.
- **`reward_norm = .centered_ranks`** (`es.Config`): Salimans-style
  centered-rank fitness shaping — coefficients are rank/(N-1) - 0.5 in
  [-0.5, 0.5], invariant to monotone reward transforms, outliers clamped by
  construction. Ties break by member index (pinned, deterministic — a
  checkpoint contract; numpy argsort tie order is implementation-defined).
  Trade-off vs z_score: all-equal rewards still take a mean-zero phantom
  step (z_score self-stops exactly), so pair long rank-shaped runs with
  `--save-every` + eval-based selection.

Neither normalization shrinks the step near a sharp optimum (both rescale to
a fixed spread every iteration), so the practical brakes remain: bounded
saturating rewards, conservative sigma, and eval-selected checkpoints.

**LLM fine-tuning.** `zig build es-finetune` (examples/es_finetune.zig) is
finetune.zig's gradient-free twin — same dataset plumbing, same trainer
forward, same checkpoint layout, so runs compare apples-to-apples.
`--mode lora` perturbs only the q/v adapters; `--mode full` perturbs every
resident float weight of the base model (needs an f32/f16/bf16 GGUF — 
quantized blocks cannot take noise; transcode with export-gguf). Rewards
(`--reward`), plus `--norm z_score|centered_ranks`:

- `rule`: DeepSeek-R1-style rule reward on greedy generations (unigram-F1
  accuracy + 0.1 * response-envelope format) — bounded and saturating, but
  generation-bound on CPU.
- `acc` (recommended for loss-style training): the bounded teacher-forced
  composite `token_accuracy + 0.1 * exp(-mean CE)` in [0, ~1.1] — one
  forward per sample like nll, the same accuracy-dominant shape as `rule`,
  dense (the likelihood term breaks ties, so no interior stalls at small
  sigma), and softly self-stopping as the population saturates.
- `nll`: raw negative mean CE — directly comparable with finetune.zig's
  loss curve, but UNBOUNDED: expect the outlier pathology above on long
  runs unless paired with `--norm centered_ranks` and checkpoint selection. Measured on M1 Max (Qwen3-0.6B, ReleaseFast): lora +
nll ≈ 0.9 s per member-eval (population 8 x batch 5 ≈ 7-10 s/iteration);
full mode adds the noise-regeneration cost over 0.6 B parameters
(≈ 30 s/iteration at population 4, batch 1); rule rewards are
generation-bound. ES needs MANY more iterations than backprop for the same
movement — the paper runs 300-500 iterations at population 30 — so treat the
demo defaults as a mechanism showcase, not a convergence recipe.

**Anchored weight decay (AWD).** The third opt-in extension
(`anchor_decay = .l1|.l2` + `anchor_lambda`; arXiv:2605.30148, the same
group's anti-forgetting follow-up): after each ES update, a proximal step
pulls theta toward a fixed anchor captured by `captureAnchor` — l2 shrinks
`theta - theta_0` by `(1 - alpha*lambda)`, l1 soft-thresholds it at
`alpha*lambda` (exact zeroing, induces sparsity in the fine-tuning delta).
It counteracts the accumulated random-walk drift in reward-irrelevant
directions — the paper's cheap substitute for a large population, and the
principled brake on the post-peak drift described above. Reference values:
l2 lambda = 10, l1 lambda = 0.01 at alpha = 5e-4; their tuning heuristic is
to start high and lower lambda until target-task performance stops
suffering. Fine-tuning only: anchoring to a random init pins the model to
noise. The anchor is never serialized — on resume, capture it from the
reconstructed initial weights BEFORE loading the checkpoint (es_finetune
orders this correctly; `es_anchor_decay`/`es_anchor_lambda` persist the
configuration).

**From-scratch acceptance demo.** `zig build es-spirals` trains
spirals.zig's exact MLP on the exact two-spirals data FROM RANDOM INIT with
pure ES (no backward anywhere) and self-verifies: 100% accuracy / CE -> 0 in
~15k iterations (~75 s ReleaseFast, ~5 ms/iteration), population 128
evaluated member-parallel — each worker thread owns a full replica +
ExecContext, `materializeMember` fills it, only scalar rewards cross
threads. Fine-tuning starts near a solution; this run proves the method
OPTIMIZES. Note the shaping reversal: on this smooth landscape with
well-behaved bounded rewards, z-score converges and `centered_ranks` stalls
— rank shaping trades magnitude information for outlier immunity, so pick
by reward regime, not by habit.

ES pitfalls:

- Evaluating a member without `restore` → the next perturb errors
  (`MemberActive` tripwire); on an eval error mid-`step` the driver restores
  before propagating.
- Changing `population`, `seed`, or the noise scheme mid-run breaks the
  (seed, iteration) → noise contract exactly like editing an optimizer
  checkpoint; the es_finetune resume takes all three from the checkpoint.
- `--mode full` on a quantized GGUF → rejected up front
  (`QuantizedWeightsUnsupported` with the transcode hint).
- Reward != finite (NaN/Inf) poisons the z-score for the whole iteration —
  clamp or zero pathological rewards in the evaluator (the reference gives
  timeout/exception rollouts reward 0.0).

**Ternary-native ES (training = inference).** Alongside the float slots,
`es.Trainer` accepts packed TQ2_0 ternary genomes (`addTernaryParam`):
perturbation is sparse trit flips regenerated from the dedicated `es_trits`
counter-RNG domain (antithetic pairs mirror deltas; restore replays a sparse
undo log — clamping is lossy), and the update is EGGROLL-style
vote-and-threshold: fitness-weighted votes on touched indices, top-K by
|vote| move one bin toward sign(vote), clamped, with
`K = round(ternary_update_fraction·len / (1 + ternary_update_decay·t))`.
Weights never leave {-1, 0, +1}·d, so members evaluate through the real
TQ2_0 int8 kernels and the trained state is byte-for-byte the served model.
The three `ternary_*` knobs are checkpoint contracts (persisted as
`es_ternary_*`). `zig build es-ternary-spirals` is the from-scratch
acceptance demo. Full design record: `docs/TERNARY.md`.
