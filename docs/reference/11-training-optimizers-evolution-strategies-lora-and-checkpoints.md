# 11. Training: optimizers, evolution strategies, LoRA, and checkpoints

Training on the eager runtime is five modules, all reachable from the root
export: `fucina.optim` (gradient-descent optimizers, param groups, LR
schedules, clipping, optimizer-state persistence), `fucina.es`
(gradient-free evolution strategies), `fucina.lora` (LoRA adapters over
frozen linears), `fucina.ParamRegistry` + `fucina.state_dict` (named
parameter collection and safetensors state dicts), and
`fucina.training_checkpoint` (the resumable checkpoint directory layout).
The contract document is [TRAINING.md](../TRAINING.md); everything below is
exercised end-to-end by `zig build spirals`, `zig build es-spirals`,
`zig build finetune`, and `zig build es-finetune` (`examples/`, `apps/`). Autograd
semantics are [§5](05-automatic-differentiation.md); the exec-scope memory model the training loop leans on is
[§6](06-the-execution-runtime-execcontext-and-the-memory-model.md). Snippets in this section assume
`const std = @import("std"); const fucina = @import("fucina"); const optim = fucina.optim;`.

## 11.1 The shape of a training step

Training differs from inference in exactly one rule: every tensor on the
path from the parameters to the loss must stay alive until `backward()`
returns ([§5](05-automatic-differentiation.md) — `GradState` is single-owner, and consumers hold raw pointers
into the graph). Exec scopes ([§6.3](06-the-execution-runtime-execcontext-and-the-memory-model.md#63-exec-scopes-implicit-ownership-for-training-srcexeczig-srcexecruntimezig)) make that rule implicit: open a scope
around the step, write the forward in the ordinary deinit-ASAP style (deinit
on scope-owned results is a safe no-op), and close the scope after the
optimizer step. The canonical step order is `backward` → `clipGradNorm` →
`step` → `zeroGrad`:

```zig
test "one training step: forward, backward, clip, step, zero" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var w = try fucina.Tensor(.{ .class, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.1, -0.2, 0.3, 0.4 });
    defer w.deinit();
    var b = try fucina.Tensor(.{.class}).variableFromSlice(&ctx, .{2}, &.{ 0, 0 });
    defer b.deinit();
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, -1, 0.5 });
    defer x.deinit();
    const labels = [_]usize{ 0, 1 };

    var opt = optim.AdamW.init(alloc, .{ .lr = 0.05, .weight_decay = 0.01 });
    defer opt.deinit();
    try opt.addParam(&w); // params must outlive the optimizer
    try opt.addParam(&b);

    var first: f32 = 0;
    var last: f32 = 0;
    for (0..20) |i| {
        const scope = ctx.openExecScope(); // the scope owns the step's graph
        defer ctx.closeExecScope(scope);
        const z = try x.dot(&ctx, &w, .in);
        const logits = try z.add(&ctx, &b);
        const loss = try logits.crossEntropy(&ctx, .class, &labels, .{});
        try loss.backward(&ctx);
        _ = try opt.clipGradNorm(&ctx, 1.0); // after backward, before step
        try opt.step(&ctx);
        opt.zeroGrad();
        if (i == 0) first = try loss.item();
        last = try loss.item();
    }
    try std.testing.expect(last < first);
}
```

Gradient accumulation needs no extra machinery: `backward()` ADDS into each
parameter's persisted gradient, leaf gradients live outside exec scopes,
`step()` reads them non-destructively, and `zeroGrad()` is the only clear.
N micro-batches + one `step()` is the accumulation recipe; scale the LOSS
(not the gradients) by the window normalizer, clip once after the window,
key LR schedules by the macro step, and checkpoint only at window
boundaries (accumulated gradients are never serialized). The full recipe,
its normalization arms, and its determinism contract are
[TRAINING.md](../TRAINING.md) [§4](04-tensor-operations.md).

## 11.2 Optimizers (`src/optim.zig`)

`src/optim.zig` is the facade (every `fucina.optim.*` name below); the
bodies live in `src/optim/` — `common` (the shared substrate: `Param`,
state-buffer dtypes, the deterministic norm reduction, clipping), `frame`
(checkpoint frames, [§11.5](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#115-optimizer-state-persistence-fzt1-snapshots-vs-named-state-dicts-srcoptimzig)), `moment_pair` (Adam/AdamW), `muon`, `apollo`,
`sgd`, `schedule` ([§11.4](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#114-gradient-clipping-and-lr-schedules-srcoptimzig)), and `set` ([§11.3](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#113-param-groups-optimizerset-srcoptimzig)).

Each optimizer is a faithful port of a reference implementation, pinned by
golden parity tests against the actual references (PyTorch 2.12, Keller
Jordan's muon.py, apollo_torch — `src/optim_tests.zig`):

| Type | Config | Reference | State per param element |
|---|---|---|---|
| `optim.SGD` | `SgdConfig` | `torch.optim.SGD` (single-tensor) | 0 B; 4 B with momentum (2 B bf16) |
| `optim.Adam` | `AdamConfig` | `torch.optim.Adam` (coupled L2 decay) | 8 B m+v (down to 4 B bf16) |
| `optim.AdamW` | `AdamWConfig` | `torch.optim.AdamW` (decoupled decay) | 8 B m+v (down to 4 B bf16) |
| `optim.Muon` | `MuonConfig` | Keller Jordan reference + Moonlight scale | 4 B momentum (2 B bf16) + embedded AdamW fallback |
| `optim.Apollo` | `ApolloConfig` | apollo_torch (arXiv 2412.05270) | ~`8·rank·max(dim)` B moments + `4·rank·min(dim)` B resident projection per matrix (always f32) |

All five share one surface (`Muon`/`Apollo` add the fallback registrars):

```zig
pub fn init(allocator: Allocator, config: Config) Self      // SGD panics here on bad nesterov
pub fn deinit(self: *Self) void                             // frees slots + state; params stay caller-owned
pub fn addParam(self: *Self, t: anytype) !void              // t: pointer to an f32/f16/bf16 autograd variable
pub fn addParamNamed(self: *Self, t: anytype, name: []const u8) !void
pub fn addFallbackParam(self: *Self, t: anytype) !void      // Muon/Apollo only
pub fn addFallbackParamNamed(self: *Self, t: anytype, name: []const u8) !void
pub fn step(self: *Self, ctx: *ExecContext) !void
pub fn zeroGrad(self: *Self) void
pub fn clipGradNorm(self: *Self, ctx: *ExecContext, max_norm: f32) !f32
pub fn gradSquaredNorm(self: *Self, ctx: *ExecContext) !f64
pub fn scaleGradients(self: *Self, ctx: *ExecContext, factor: f32) !void
pub fn saveState(self: *const Self, writer: *std.Io.Writer) !void
pub fn loadState(self: *Self, reader: *std.Io.Reader) !void
pub fn collectGradStates(self: *const Self, set: *GradStateSet, allocator: Allocator) !void // OptimizerSet plumbing (private set type)
```

**Ownership.** `addParam` goes through `optim.Param.of`: it retains a
refcounted view of the variable's storage plus the raw `*GradState` pointer,
so the facade struct may move by value but the parameter must OUTLIVE the
optimizer (the tensor owns the GradState).

**16-bit params and f32 master weights.** `addParam` also accepts f16/bf16
variables ([§5.1](05-automatic-differentiation.md#51-the-gradient-model-srcagtensorzig-srcagcorezig): their gradients are f32). Each 16-bit slot allocates an
optimizer-owned f32 MASTER copy at registration (widened once from the
param values): every update kernel steps the master — so the step math is
identical to the f32 path, and updates below 16-bit resolution accumulate
instead of rounding away — and the master is narrowed back into the 16-bit
storage after each step. Ordering contract: load parameter values BEFORE
registering the param (or restore via `loadState`, whose v5 frames carry
the master); a value load after registration leaves the master stale.

```zig
test "bf16 params train through f32 masters" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const W = fucina.Tensor(.{ .dtype = .bf16, .tags = .{ .out, .in } });
    var w = try W.variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0x3f80, 0xc000, 0x3f00, 0x4040 }); // 1, -2, 0.5, 3
    defer w.deinit();
    var x = try fucina.Tensor(.{ .t, .in }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 2 });
    defer x.deinit();

    var opt = fucina.optim.AdamW.init(alloc, .{ .lr = 0.05 });
    defer opt.deinit();
    try opt.addParam(&w); // allocates + fills the f32 master

    const scope = ctx.openExecScope();
    var y = try x.dot(&ctx, &w, .in); // native mixed GEMM; dW arrives as f32
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    ctx.closeExecScope(scope);

    try opt.step(&ctx); // steps the master, narrows back into w's storage
    opt.zeroGrad();
    try std.testing.expect(w.requiresGrad());
    try std.testing.expect((try w.grad(&ctx)) == null); // cleared
}
```

Errors from `addParam`: `error.NotAVariable`
(constant or grad-free tensor), `error.NonContiguousParam`,
`error.DuplicateParam` (same variable twice in ONE instance — registering it
with two *different* instances is undetectable per-instance and silently
double-steps; `OptimizerSet.add` closes that gap, [§11.3](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#113-param-groups-optimizerset-srcoptimzig)). Names passed to
`addParamNamed` are BORROWED and must outlive the optimizer (string
literals and model-struct fields qualify). Optimizer state (moments,
momentum, projections) is optimizer-owned and freed by `deinit`.

**`optim.Param`** is the type-erased slot handle (`value` — an
f32/f16/bf16 storage union, `grad_state`, `rows`, `cols`, `raw_rank`,
optional `name`, the f32 `master` for 16-bit slots; `pub fn of(t) !Param`,
`pub fn len(self) usize`). `rows`/`cols` describe the matrix view the
matrix-aware optimizers use: dim 0 by the product of the remaining dims —
Keller's conv-filter flattening `[d0, d1*d2*...]`.

**`step` semantics.** Parameters whose accumulated gradient is null are
skipped (PyTorch behavior); a gradient whose element count disagrees with
the parameter is `error.GradShapeMismatch`. Elementwise updates are fused
single-pass, element-independent maps chunked over the worker pool at
2^17 elements and above — bitwise identical to the serial loop for any thread count
(reductions such as norms run over a fixed chunk grid with a pinned
combine order, equally thread-count-invariant). Scalar prep
(bias corrections, step sizes) runs in f64 and rounds once to f32, matching
torch's Python-float scalars to within a few f32 ulps.

**`optim.StateDType`** (`enum(u8) { f32 = 0, bf16 = 1 }`; the values are v4
checkpoint wire tags — never renumbered). Every elementwise optimizer can
store its moment/momentum buffers in bf16 via `state_dtype` (and, for
Adam/AdamW, the separate `second_moment_dtype`): step math stays f32
(widen-on-read, NaN-guarded round-to-nearest-even narrow-on-write), updates
stay element-independent, and since the step is memory-bound the narrower
state is measurably *faster* (bench: `zig build bench-optim`). First
moments/momentum tolerate bf16 well; AdamW/Adam `v` is precision-sensitive
(with beta2 = 0.999 its ~0.1 %/step change sits below bf16's ~0.39 %
resolution, so the EMA can stall) — hence the separate opt-in. The f32
default keeps every existing checkpoint byte-identical.

### Config structs and defaults

```zig
pub const SgdConfig = struct {
    lr: f32 = 1e-3,
    momentum: f32 = 0,          // 0 = no momentum buffer at all
    dampening: f32 = 0,
    weight_decay: f32 = 0,      // COUPLED L2 (g += wd*p), PyTorch SGD
    nesterov: bool = false,     // requires momentum > 0, dampening == 0
    state_dtype: StateDType = .f32,
};
pub const AdamConfig  = struct { lr: f32 = 1e-3, beta1: f32 = 0.9, beta2: f32 = 0.999,
    eps: f32 = 1e-8, weight_decay: f32 = 0,     // coupled decay
    state_dtype: StateDType = .f32, second_moment_dtype: StateDType = .f32 };
pub const AdamWConfig = struct { lr: f32 = 1e-3, beta1: f32 = 0.9, beta2: f32 = 0.999,
    eps: f32 = 1e-8, weight_decay: f32 = 0.01,  // decoupled, applied BEFORE the step
    state_dtype: StateDType = .f32, second_moment_dtype: StateDType = .f32 };
```

`SGD.init` **panics** (in every build mode, deliberately not a debug assert)
when `nesterov` is set with zero momentum or nonzero dampening — the PyTorch
constructor rule. With momentum, the buffer is initialized on the first step
to a clone of the first (decayed) gradient, not zeros (`buf = d_p.clone()`);
under bf16 state that clone is stored narrowed. `AdamW` applies decay to the
parameter before the moment update and adds `eps` after dividing `sqrt(v)`
by the bias correction — the exact `_single_tensor_adam` order; `Adam`
differs only in coupling the decay into the gradient.

```zig
pub const MuonScale = enum { spectral, match_rms_adamw };
pub const MuonConfig = struct {
    lr: f32 = 0.02, momentum: f32 = 0.95, nesterov: bool = true,
    ns_steps: u32 = 5, weight_decay: f32 = 0,
    scale: MuonScale = .spectral,
    state_dtype: StateDType = .f32,
    fallback: AdamWConfig = .{ .lr = 3e-4, .beta1 = 0.9, .beta2 = 0.95, .eps = 1e-10, .weight_decay = 0 },
};
```

Muon runs lerp-form momentum, Newton-Schulz-5 orthogonalization (f32; the
reference uses bf16 on GPU), and a shape-dependent scale: `.spectral` is
Keller's `sqrt(max(1, rows/cols))`, `.match_rms_adamw` is Moonlight's
`0.2*sqrt(max(rows, cols))` (reuse AdamW-tuned lr/wd). Routing: `addParam`
sends rank ≥ 2 params to the Muon path and auto-routes 0D/1D params (biases,
norms) to the embedded AdamW fallback; embeddings and output/classifier
heads are 2D but must NOT be orthogonalized — route them explicitly with
`addFallbackParam`/`addFallbackParamNamed`. Newton-Schulz transients come
from the ExecContext `BufferPool`. The iteration itself is public:

```zig
pub fn newtonSchulz5(ctx: *ExecContext, u: *const RawTensor, steps: u32) !RawTensor
```

Frobenius-normalize, then iterate the tuned quintic; when rows > cols the
iteration runs on the transpose so the Gram matrix has the small dimension.
The result approximates `U·V^T` with singular values in roughly (0.5, 1.5) —
by design. `u` must be rank-2 and is never mutated; the caller owns the
result.

```zig
pub const ApolloScaleType = enum { channel, tensor };
pub const ApolloConfig = struct {
    lr: f32 = 0.01, beta1: f32 = 0.9, beta2: f32 = 0.999,
    eps: f32 = 1e-6,                 // legacy-HF default; HF Trainer overrides to 1e-8
    weight_decay: f32 = 0,
    rank: usize = 128, update_proj_gap: u64 = 200,
    scale: f32 = 1.0, scale_type: ApolloScaleType = .channel,
    correct_bias: bool = true, scale_front: bool = false,
    disable_norm_growth_limiter: bool = false,
    seed: u64 = 0,
    pub fn mini() ApolloConfig      // rank 1, .tensor scaling, scale 128 (APOLLO-Mini)
};
```

**APOLLO rank/projection specifics.** Only rank-2 params take the low-rank
path (`addParam` auto-routes everything else to the fallback; use
`addFallbackParam` for embeddings/heads). Per matrix: the gradient is
projected into a rank-`rank` space (tall params `rows ≥ cols` project the
column space with P `(rank, cols)`, compressed grad `R = G·P^T` of shape
`(rows, rank)`; wide params project the row space, `R = P^T·G` of shape
`(rank, cols)`), AdamW moments run in the compressed space, channel- or
tensor-wise scaling factors are computed from the un-bias-corrected
`m/(sqrt(v)+eps)` vs the raw `R` (f64 accumulators, `+1e-8` division guard),
the full-size update is the raw gradient rescaled per channel, the Fira
norm-growth limiter clips per-step `||U||_F` growth at gamma 1.01 (its norm
reduction is the deterministic fixed-chunk one), and the final step is scaled SGD with
decoupled decay applied AFTER the step at the raw lr — the reference's
legacy-HF order, deliberately different from `AdamW`. The fallback path is
likewise the legacy-HF AdamW (`denom = sqrt(v) + eps`, bias correction
folded into the scalar, decay after the step).

**APOLLO RNG contract.** Projections are REGENERATED, never stored: P is a
deterministic function of `(slot seed, step / update_proj_gap)` through the
repo-owned `rng.gaussianFill` (splitmix64 + Box-Muller, §"RNG" in
[TRAINING.md](../TRAINING.md) [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)) with i.i.d. `N(0, 1/rank)` entries; the
per-param seed is `config.seed +% (1-based rank-slot index)`. The
(seed → values) mapping is a checkpoint contract — deliberately not
`std.Random`, so it survives toolchain upgrades. Regeneration uses the
pre-increment step counter (chunks `[0,T), [T,2T), ...`); moments are NOT
reset on regeneration; `loadState` restores the stored per-slot seed and
forces regeneration on the next step. Note the APOLLO recipes train with
global clipping disabled — the norm-growth limiter replaces it
(`clipGradNorm` is still provided for mixed setups).

## 11.3 Param groups: `OptimizerSet` (`src/optim.zig`)

A param group is exactly {hyperparams, params, state} — one optimizer
instance. `optim.OptimizerSet` makes N instances feel like one optimizer,
type-erased through `optim.AnyOptimizer`:

```zig
pub const AnyOptimizer = struct {
    ptr: *anyopaque, vtable: *const VTable,
    pub const VTable = struct { step, zeroGrad, gradSquaredNorm, scaleGradients, saveState, loadState };
    // pub fn step / zeroGrad / gradSquaredNorm / scaleGradients / saveState / loadState
};
pub fn anyOptimizer(opt: anytype) AnyOptimizer   // wrap a *SGD / *Adam / *AdamW / *Muon / *Apollo

pub const OptimizerSet = struct {
    pub fn init(allocator: Allocator) OptimizerSet
    pub fn deinit(self: *OptimizerSet) void       // frees only the set; members stay caller-owned
    pub fn add(self: *OptimizerSet, opt: anytype) !void
    pub fn step(self: *OptimizerSet, ctx: *ExecContext) !void
    pub fn zeroGrad(self: *OptimizerSet) void
    pub fn gradSquaredNorm(self: *OptimizerSet, ctx: *ExecContext) !f64
    pub fn scaleGradients(self: *OptimizerSet, ctx: *ExecContext, factor: f32) !void
    pub fn clipGradNorm(self: *OptimizerSet, ctx: *ExecContext, max_norm: f32) !f32 // GLOBAL norm
    pub fn saveState(self: *const OptimizerSet, writer: *std.Io.Writer) !void       // FZO3 frame
    pub fn loadState(self: *OptimizerSet, reader: *std.Io.Reader) !void
};
```

Members are BORROWED via raw pointer (they must not move or be freed while
the set is in use). `add` calls the member's `collectGradStates` to check
every parameter against all previously-added members: registering the same
variable into two groups returns `error.DuplicateParam` and leaves the set
unchanged — the cross-instance double-step the per-optimizer guard cannot
see. Mixing optimizer types in one set works (Muon for the trunk, AdamW for
an adapter). `clipGradNorm` is global across all groups — the
`clip_grad_norm_(model.parameters())` contract. `loadState` checks the
member count (`error.CheckpointShapeMismatch`) and loads members in order,
transactionally per member.

## 11.4 Gradient clipping and LR schedules (`src/optim.zig`)

**Clipping** is `torch.nn.utils.clip_grad_norm_` semantics: compute
`total = sqrt(sum ||g||^2)` over every registered param (deterministic
fixed-chunk f64 reduction); if `total > max_norm`, scale every gradient by
`max_norm / (total + 1e-6)`; return the PRE-clip norm. Call after
`backward()`, before `step()`. `scaleGradients(ctx, factor)` is the raw
primitive (also the grad-side accumulation normalizer); `gradSquaredNorm`
exposes the sum of squares for custom policies. On `Muon`/`Apollo` the norm
spans the matrix path AND the fallback.

**Schedules** are one small hook plus pure factor functions:

```zig
pub const LrSchedule = struct {
    pub fn init(allocator: Allocator) LrSchedule
    pub fn deinit(self: *LrSchedule) void
    pub fn attach(self: *LrSchedule, lr: *f32) !void   // captures *lr as the base
    pub fn apply(self: *const LrSchedule, factor: f64) void // lr = base * factor
};
pub fn warmupCosineFactor(step: u64, total_steps: u64, warmup_steps: u64, min_factor: f64) f64
```

`attach` points at the public `config.lr` field of any optimizer (and of a
Muon fallback: `&muon.fallback.config.lr`) and captures the current value as
the base — attach before the first `apply`, or the in-effect factor gets
baked into the base; re-attaching the same pointer refreshes its base
instead of duplicating the entry. The pointee must outlive the schedule.
Because the factor is a pure function of the step, resuming from a
checkpoint just re-applies it — this is why lr is deliberately NOT a
validated field of optimizer checkpoints. `warmupCosineFactor` is linear
warmup from `1/warmup_steps` to 1 over `warmup_steps` (0-based `step`), then
cosine decay to `min_factor` over the remaining steps:

```zig
test "warmupCosineFactor endpoints" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), optim.warmupCosineFactor(0, 100, 10, 0.1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), optim.warmupCosineFactor(9, 100, 10, 0.1), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1), optim.warmupCosineFactor(100, 100, 10, 0.1), 1e-9);
}
```

Groups, schedule, and clipping compose — the standard LLM recipe in
miniature (`examples/spirals/main.zig` `groupsDemo` proves this composition
resumes bit-exactly):

```zig
test "param groups under one OptimizerSet with a warmup-cosine schedule" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var w = try fucina.Tensor(.{ .class, .in }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 0.1, -0.2, 0.3, 0.4 });
    defer w.deinit();
    var b = try fucina.Tensor(.{.class}).variableFromSlice(&ctx, .{2}, &.{ 0, 0 });
    defer b.deinit();
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, -1, 0.5 });
    defer x.deinit();
    const labels = [_]usize{ 0, 1 };

    var decay = optim.AdamW.init(alloc, .{ .lr = 2e-2, .weight_decay = 0.1 });
    defer decay.deinit();
    var no_decay = optim.AdamW.init(alloc, .{ .lr = 2e-2, .weight_decay = 0 });
    defer no_decay.deinit();
    try decay.addParam(&w); // matrices: decayed group
    try no_decay.addParam(&b); // biases/norms: no-decay group

    var set = optim.OptimizerSet.init(alloc);
    defer set.deinit();
    try set.add(&decay);
    try set.add(&no_decay);

    var sched = optim.LrSchedule.init(alloc);
    defer sched.deinit();
    try sched.attach(&decay.config.lr); // captures 2e-2 as the base
    try sched.attach(&no_decay.config.lr);

    for (0..4) |step_i| {
        sched.apply(optim.warmupCosineFactor(step_i, 100, 10, 0.1));
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const logits = try (try x.dot(&ctx, &w, .in)).add(&ctx, &b);
        const loss = try logits.crossEntropy(&ctx, .class, &labels, .{});
        try loss.backward(&ctx);
        _ = try set.clipGradNorm(&ctx, 1.0); // GLOBAL norm across both groups
        try set.step(&ctx);
        set.zeroGrad();
    }
    // Linear warmup: factor at step 3 of a 10-step warmup is 4/10.
    try std.testing.expectApproxEqAbs(@as(f32, 2e-2 * 0.4), decay.config.lr, 1e-9);
}
```

## 11.5 Optimizer-state persistence: FZT1 snapshots vs named state dicts (`src/optim.zig`)

Parameter values have two formats; optimizer internals a third.

**Positional FZT1 (legacy).** f32-only, order-based (stream magic `FZT1`;
layout in [§12.7](12-model-io-gguf-and-safetensors.md#127-training-checkpoint-directory-and-native-optimizer-frames-srctraining_checkpointzig-srcoptimzig)):

```zig
pub fn saveTensors(writer: *std.Io.Writer, tensors: anytype) !void
pub fn loadTensors(allocator: Allocator, reader: *std.Io.Reader, tensors: anytype) !void
```

`tensors` is a tuple of pointers to contiguous f32 facade tensors (variables
or constants). The loading program must list the same tensors in the same
order; shapes are validated (`error.CheckpointShapeMismatch`), and the load
is transactional — the whole stream is staged and validated before any
tensor is written, so a truncated or mismatched stream leaves every
destination byte-unchanged. Use it only for closed-world snapshots where the
saving and loading code are the same program; new code should prefer the
named form.

**Named, dtype-aware state dicts.** Re-exported from `fucina.state_dict`
([§11.7](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#117-state-dicts-srcstate_dictzig)) for convenience: `optim.NamedTensor`, `optim.NamedTensorMut`,
`optim.LoadOptions`, `optim.saveStateDict`, `optim.loadStateDict`,
`optim.loadStateDictFromFile`. Entries
carry a unique name, dtype (f32/f16/bf16/i64, raw byte passthrough), shape,
and bytes; the wire format is a valid safetensors file; the load matches
stream entries BY NAME so entry order is free. This is the portable format — it is
what `model.safetensors`/`adapters.safetensors` in a checkpoint directory
contain, and any safetensors consumer can read it.

**Optimizer internals** — moments, momentum, per-slot step counters, APOLLO
seeds/limiter memory — serialize through each optimizer's
`saveState`/`loadState` (and `OptimizerSet`'s, which concatenates member
frames under `FZO3`). Frame magics:

| Optimizer | all-f32 state (v3) | any bf16 state (v4) | any 16-bit param (v5) |
|---|---|---|---|
| Adam | `FZAD` | `FZD4` | `FZD5` |
| AdamW | `FZA3` | `FZA4` | `FZA5` |
| Muon | `FZM3` (fallback frames follow) | `FZM4` | `FZM5` |
| SGD | `FZS3` | `FZS4` | `FZS5` |
| APOLLO | `FZP3` (state always f32) | — | `FZP5` |

v5 frames additionally persist each 16-bit slot's f32 MASTER weights
(per-slot presence flag + raw f32 bytes): resuming from the narrowed
values instead would re-round the master and lose the sub-16-bit update
accumulation. `loadState` installs the checkpoint master and narrows it
into the param storage; when a v3/v4 frame (or a v5 slot without a master)
loads into a 16-bit slot, the master re-widens from the current param
values instead.

Writers emit v3 whenever every state buffer is f32 — byte-identical to
pre-`StateDType` builds, so older builds keep reading new f32 checkpoints —
and v4 otherwise (identical layout except each state buffer is prefixed by
one u8 `StateDType` tag). Loaders accept both and require the stored dtype
to match the configured one EXACTLY (`error.CheckpointDtypeMismatch`; v3
implies f32 everywhere; an unknown tag is
`error.CheckpointUnsupportedDtype`). There is deliberately no implicit
f32↔bf16 conversion: it would silently break bit-exact resume.

Load-time validation: 4-byte magic (`error.CheckpointMagicMismatch`),
structural config fields (`error.CheckpointConfigMismatch` — Muon validates
scale/nesterov/ns_steps; SGD momentum/dampening/nesterov; Apollo
rank/update_proj_gap/scale/scale_type/correct_bias/scale_front/limiter
flag; lr is deliberately NOT validated — schedules legitimately change it),
per-slot dims (`error.CheckpointShapeMismatch`). Slots are matched BY NAME:
the explicit `addParamNamed` name, else the auto-name `"param<i>"` from the
slot's index within its slot list (Muon/Apollo fallback lists number
independently). Named params may therefore re-register in ANY order within
their list; unnamed params must keep their **absolute slot indices** to
reproduce their auto-names — the auto-name is `"param<i>"` from the slot's
position in the whole list, so reordering *named* entries around an unnamed
one also breaks it (`error.CheckpointUnknownName` on load). Name errors:
`error.CheckpointInvalidName`
(empty/too long/NUL/invalid UTF-8), `error.CheckpointDuplicateName` (also
raised at SAVE time when an explicit name collides with an auto-name),
`error.CheckpointUnknownName` (stream record matches no slot),
`error.CheckpointMissingEntry` (a slot left unfilled).

Every `loadState`/`loadTensors` is transactional per optimizer instance:
records decode into scratch and the whole stream validates before the first
live byte is written, so a bad stream is a no-op (`Muon` commits its own
slots only after its embedded fallback loads, keeping the pair atomic;
`OptimizerSet` is transactional per member — treat a failed set load as
fatal for the whole run). **Bit-exact resume**: state restore is byte-exact
and updates are thread-count-invariant, so resume replays bit-exactly as
long as the surrounding forward/backward is deterministic (the one caveat:
a tensor fed by 3+ heavy async backward branches accumulates in completion
order — [TRAINING.md](../TRAINING.md) [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)). `zig build spirals` asserts
bit-identical final parameters after a halfway-checkpoint resume for SGD,
AdamW, Muon, APOLLO, and APOLLO-Mini, and for the groups+schedule+clip
combo (plain `Adam`'s FZAD/FZD4 resume path is not covered by that gate).

```zig
test "optimizer state: name-matched slots round-trip; structural config is validated" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();
    var w = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer w.deinit();
    var b = try fucina.Tensor(.{.e}).variableFromSlice(&ctx, .{2}, &.{ 5, 6 });
    defer b.deinit();

    var opt = optim.SGD.init(alloc, .{ .lr = 0.1, .momentum = 0.9 });
    defer opt.deinit();
    try opt.addParamNamed(&w, "w");
    try opt.addParamNamed(&b, "b");
    {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope);
        const loss = try w.sumAll(&ctx);
        try loss.backward(&ctx);
        try opt.step(&ctx); // populates the momentum buffer
        opt.zeroGrad();
    }
    var buf: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try opt.saveState(&writer);

    // Resume: named slots may re-register in any order.
    var opt2 = optim.SGD.init(alloc, .{ .lr = 0.1, .momentum = 0.9 });
    defer opt2.deinit();
    try opt2.addParamNamed(&b, "b");
    try opt2.addParamNamed(&w, "w");
    var reader = std.Io.Reader.fixed(writer.buffered());
    try opt2.loadState(&reader);

    // Structural config fields must match the stored ones (lr is not one).
    var opt3 = optim.SGD.init(alloc, .{ .lr = 0.1, .momentum = 0.5 });
    defer opt3.deinit();
    var reader3 = std.Io.Reader.fixed(writer.buffered());
    try std.testing.expectError(error.CheckpointConfigMismatch, opt3.loadState(&reader3));
}
```

**When to use which.** FZT1 `saveTensors` for quick same-program f32
snapshots; named state dicts for anything that must survive refactoring,
mixed dtypes, or foreign consumers (they are plain safetensors);
`saveState`/`loadState` alongside the state dict whenever training must
RESUME (moments and step counts are not reconstructible). The complete
error set is `optim.OptimError`.

## 11.6 `ParamRegistry` (`src/param_registry.zig`)

`fucina.ParamRegistry` is the named-parameter seam between models,
checkpoints, and trainers. It owns no model tensors: it BORROWS named
f32/f16/bf16 (and, via explicit `addParam` only, frozen i64) facade
tensors, retaining refcounted storage views (dtype-erased), so the
original tensors and their GradStates must outlive the registry and any
optimizer it registers into. Names are COPIED (registry-owned).

```zig
pub const ParamRegistry = struct {
    pub fn init(allocator: Allocator) ParamRegistry
    pub fn deinit(self: *ParamRegistry) void
    pub fn addParam(self: *ParamRegistry, name: []const u8, t: anytype) !void
    pub fn collect(self: *ParamRegistry, model: anytype) !void
    pub fn collectPrefixed(self: *ParamRegistry, prefix: []const u8, model: anytype) !void
    pub fn parameterCount(self: *const ParamRegistry) usize
    pub fn view(self: *const ParamRegistry, index: usize) ParamView
    pub fn zeroGrad(self: *ParamRegistry) void
    pub fn addParamsTo(self: *const ParamRegistry, opt: anytype) !void
    pub fn saveStateDict(self: *const ParamRegistry, writer: *std.Io.Writer) !void
    pub fn loadStateDict(self: *ParamRegistry, reader: *std.Io.Reader, options: state_dict.LoadOptions) !void
    pub fn loadStateDictFromFile(self: *ParamRegistry, file: *const safetensors.File, options: state_dict.LoadOptions) !void
};
pub const ParamView = struct { name, dtype, shape, bytes: []u8, trainable: bool }; // also `fucina.ParamView`
```

- `addParam` registers one tensor under an explicit name. Variables
  (`grad_state != null`) are trainable; constants and grad-free typed
  f16/bf16/i64 tensors register as FROZEN entries — saved and loaded, but
  skipped by `addParamsTo` and `zeroGrad` (i64 is for frozen integer
  metadata riding a checkpoint, e.g. the cartridge `draft_reference`
  token ids, [§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig)). Errors: `CheckpointInvalidName`,
  `CheckpointDuplicateName`, `error.NonContiguousParam`; unsupported
  dtypes are a compile error.
- `collect` reflectively registers every f32/f16/bf16 tensor field of a
  model (mutable struct pointer) — deliberately NOT i64, so adding an
  integer tensor field can never silently grow a collected model's
  checkpoint schema — naming by field path: nested structs get
  dotted names (`"encoder.weight"`), arrays and slices index with dots
  (`"layers.0.weight"`), mutable single-item pointers are followed
  transparently, optionals descend into the payload, and tagged unions
  descend into the ACTIVE arm under the same prefix (exactly one arm is
  live, so the checkpoint path stays stable across storage-variant arms —
  e.g. an f16 vs bf16 weight union). Const pointers, comptime fields,
  scalars, and unsupported-dtype tensors are skipped. `collectPrefixed`
  prepends a root prefix.
- `view(i)` returns a borrowed per-entry view in registration order —
  `bytes` aliases the live (mutable) storage; this is the seam
  gradient-free consumers use (`es.Trainer.addRegistry`, [§11.11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#1111-evolution-strategies-srceszig)), frozen
  entries included.
- `addParamsTo(opt)` forwards each TRAINABLE entry (f32/f16/bf16) to
  `opt.addParamNamed(&param, name)` — so trainers delegate registration and
  checkpoint identity to the registry in one call, and optimizer slot names
  automatically equal the state-dict paths.
- `saveStateDict`/`loadStateDict`/`loadStateDictFromFile` wrap
  `fucina.state_dict` over the full entry set (frozen included). The stream
  load stages one frame in RAM; the file load copies out of a parsed
  `safetensors.File`, so a full-weight registry resumes from a
  `File.loadMmap` mapping without a second model of resident memory (the
  `es-finetune --mode full` resume path).

**Names are the on-disk schema.** A registered name is a checkpoint field
path: strict loading matches by exact name, so RENAMING a parameter path
silently orphans old checkpoints (`CheckpointUnknownName` for the stream
entry, `CheckpointMissingEntry` for the renamed destination). When a rename
is unavoidable, do NOT loosen `strict` — pass a
`state_dict.LoadOptions.aliases` rule
(`.{ .old = "enc.w", .new = "encoder.w" }`) so old checkpoints load into
the new path while keeping the one-to-one guarantee.

Round trip through a directory (the trainable model saves; a gradient-free
constant twin loads — spirals' inference phase):

```zig
test "ParamRegistry: collect, save to a directory, load by name" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const Model = struct { w: fucina.Tensor(.{ .out, .in }) };
    var model = Model{ .w = try fucina.Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 }) };
    defer model.w.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    {
        var registry = fucina.ParamRegistry.init(alloc);
        defer registry.deinit();
        try registry.collect(&model); // names by field path: "w"
        var file = try tmp.dir.createFile(io, "model.safetensors", .{});
        defer file.close(io);
        var fbuf: [4096]u8 = undefined;
        var writer = file.writer(io, &fbuf);
        try registry.saveStateDict(&writer.interface);
        try writer.interface.flush();
    }
    // Inference twin: a CONSTANT registers as a frozen entry and still loads.
    var restored = Model{ .w = try fucina.Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 2, 3 }, &.{ 0, 0, 0, 0, 0, 0 }) };
    defer restored.w.deinit();
    {
        var registry = fucina.ParamRegistry.init(alloc);
        defer registry.deinit();
        try registry.collect(&restored);
        var file = try tmp.dir.openFile(io, "model.safetensors", .{});
        defer file.close(io);
        var fbuf: [4096]u8 = undefined;
        var reader = file.reader(io, &fbuf);
        try registry.loadStateDict(&reader.interface, .{});
    }
    try std.testing.expectEqualSlices(f32, try model.w.dataConst(), try restored.w.dataConst());
}
```

## 11.7 State dicts (`src/state_dict.zig`)

`fucina.state_dict` is the neutral named-tensor serialization layer: models,
LoRA adapters, and optimizers all speak in named entries without depending
on each other. The wire format is Hugging Face safetensors ([§12](12-model-io-gguf-and-safetensors.md)); a state
dict written here is a valid standalone safetensors file.

```zig
pub const NamedTensor    = struct { name, dtype, shape, bytes: []const u8;
    pub fn of(name: []const u8, t: anytype) !NamedTensor };     // borrowed name + storage
pub const NamedTensorMut = struct { name, dtype, shape, bytes: []u8;
    pub fn of(name: []const u8, t: anytype) !NamedTensorMut };  // requires a mutable tensor pointer
pub const Alias = struct { old: []const u8, new: []const u8 };
pub const LoadOptions = struct { strict: bool = true, aliases: []const Alias = &.{} };
pub fn saveStateDict(allocator, writer: *std.Io.Writer, entries: []const NamedTensor) !void
pub fn loadStateDict(allocator, reader: *std.Io.Reader, entries: []const NamedTensorMut, options: LoadOptions) !void
pub fn loadStateDictFromFile(allocator, file: *const safetensors.File, entries: []const NamedTensorMut, options: LoadOptions) !void
```

`NamedTensor.of` accepts a pointer to any contiguous f32/f16/bf16/i64
facade tensor (variable or constant; other dtypes are compile errors;
non-contiguous is `error.NonContiguousParam`). `NamedTensorMut.of`
additionally requires a mutable tensor pointer — a `*const` argument is a
compile error. Both the name and the storage are BORROWED — they must
outlive the entry.

`saveStateDict` validates everything before writing a byte: names must be
non-empty, NUL-free, valid UTF-8, not `"__metadata__"`, and unique; entry
byte lengths must match dtype×shape; a hand-built `NamedTensor` whose dtype
is outside f32/f16/bf16/i64 fails with `CheckpointUnsupportedDtype`. Raw bytes
are then written — no conversion. `loadStateDict` reads one safetensors
prefix from the reader and matches stream entries to destinations BY NAME
(any order), after applying the `aliases` remap to each STREAM name (first
matching rule wins). Shape and dtype must match the destination exactly
(`CheckpointShapeMismatch`/`CheckpointDtypeMismatch` — no conversion is
ever performed). Strict mode (the default) demands a one-to-one match: an
unmatched stream entry is `CheckpointUnknownName`, an unfilled destination
`CheckpointMissingEntry`. Non-strict skips unknown STREAM entries and
leaves destinations absent from the stream unchanged. The load is two-pass
transactional: every stream entry validates against its destination before
any destination byte is written; pass 2 commits with plain `@memcpy`s, so
any error leaves every destination byte-unchanged. Error set:
`state_dict.Error` (the same checkpoint error names as `optim.OptimError`).

`loadStateDict` stages the whole frame in RAM (`readPrefix`) before the
copy pass, so its peak is one payload on top of the destinations.
`loadStateDictFromFile` is the same load over an already-parsed
`safetensors.File`: it validates from the header and copies out of the
file's tensor bytes, with identical matching, remap, strictness, and
transactional behavior. Over a `File.loadMmap` file the pages stream from
the mapping as they are copied, which is how a full-model checkpoint (a
payload the size of the model) resumes at one model of resident memory.

```zig
test "state_dict: named save/load round-trip" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var w = try fucina.Tensor(.{ .out, .in }).constant(&ctx, try ctx.fromSlice(.f32, &.{ 2, 2 }, &.{ 3, -1, 4, 1 }));
    defer w.deinit();

    var buf: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try fucina.state_dict.saveStateDict(alloc, &writer, &.{
        try fucina.state_dict.NamedTensor.of("enc.w", &w),
    });

    var dst = try fucina.Tensor(.{ .out, .in }).constant(&ctx, try ctx.zeros(.f32, &.{ 2, 2 }));
    defer dst.deinit();
    var reader = std.Io.Reader.fixed(writer.buffered());
    try fucina.state_dict.loadStateDict(alloc, &reader, &.{
        try fucina.state_dict.NamedTensorMut.of("enc.w", &dst),
    }, .{});
    try std.testing.expectEqualSlices(f32, try w.dataConst(), try dst.dataConst());
}
```

## 11.8 safetensors read/write surface (`src/safetensors.zig`)

`fucina.state_dict` sits on `fucina.safetensors`, which is also usable
directly: `File` (`parse`, `parseOwned`, `load`, `loadMmap`, `deinit`,
`tensor`, `maybeTensor`, `names`, `tensorNames`, `len`, `isEmpty`),
`Sharded` (`load`, `loadMmap`, `deinit`, `tensor`, `maybeTensor`,
`tensorNames`, `len`, `isEmpty`, `total_size` — the Hugging Face
`*.safetensors.index.json` multi-file layout; `isIndexPath` recognizes it),
`TensorInfo` (+ `sliceBytesAlloc` with `TensorInfo.Slice` ranges),
`readPrefix` (one safetensors payload from a stream — what `loadStateDict`
uses; `loadStateDictFromFile` takes a `File` directly), `serialize` /
`serializeAlloc` / `saveFileAtomic`, `Tensor`,
`MetadataEntry`, `DType` (+ `bitsize`, `string`), `dtypeFromFucina` /
`dtypeToFucina`, `max_header_size`, `Error`. Container format, dtype
coverage, and mmap semantics are [§12](12-model-io-gguf-and-safetensors.md).

## 11.9 Checkpoint directories (`src/training_checkpoint.zig`)

Canonical resumable checkpoints are DIRECTORIES: the portable tensor
artifact is a clean safetensors file, Fucina-only resume state lives beside
it, and a small JSON sentinel commits the whole thing.

```text
checkpoint/
  model.safetensors        # or adapters.safetensors (LoRA runs)
  optimizer.fucina         # native optimizer frames (§11.5)
  trainer_state.json       # written LAST; the commit sentinel
```

```zig
pub const model_state_file     = "model.safetensors";
pub const adapters_state_file  = "adapters.safetensors";
pub const optimizer_state_file = "optimizer.fucina";
pub const trainer_state_file   = "trainer_state.json";
pub const Error = error{ InvalidTrainerState, UnsupportedTrainerStateVersion };

pub fn pathJoin(allocator, dir_path: []const u8, leaf: []const u8) ![]u8
pub fn beginSave(allocator, io: std.Io, dir_path: []const u8) !void
pub fn writeFileAtomic(io: std.Io, path: []const u8, context: anytype,
    comptime writeFn: fn (@TypeOf(context), *std.Io.Writer) anyerror!void) !void
pub fn saveTrainerState(allocator, io: std.Io, dir_path: []const u8, state: anytype) !void
pub fn loadTrainerState(comptime State: type, allocator, io: std.Io, dir_path: []const u8) !State
pub fn writeTrainerStateJson(state: anytype, writer: *std.Io.Writer) !void
pub fn parseTrainerState(comptime State: type, allocator, bytes: []const u8) !State
```

The trainer-state codec is generic over the caller's state struct: a
`version: u32`/`step: u64`/`seed: u64` header plus optional `?u64`/`?f64`
fields, with the writer and parser bodies comptime-generated from the
struct (JSON key = field name, emission order = declaration order). The
LLM trainers' concrete struct is `fucina_models.train.trainer_state.TrainerState`:

```zig
pub const TrainerState = struct {   // fucina_models.train.trainer_state
    version: u32 = 1, step: u64 = 0, seed: u64 = 0,
    lora_rank: ?u64, lora_alpha: ?f64, lora_dropout_p: ?f64, learning_rate: ?f64,
    accum_steps: ?u64,                       // window size; step % accum_steps == 0 at save
    data_seed: ?u64, data_epoch: ?u64, data_index: ?u64,   // models.text.data.Loader.State
    es_sigma: ?f64, es_alpha: ?f64, es_population: ?u64,
    es_noise: ?u64,                          // STABLE mapping: 0 = iid, 1 = correlated
    es_antithetic: ?u64,                     // 1 = mirrored pairs
    es_anchor_decay: ?u64, es_anchor_lambda: ?f64,          // 0/absent none, 1 l1, 2 l2
    es_iteration: ?u64,
};
```

**Crash-consistency protocol.** `beginSave` creates the directory and
DELETES `trainer_state.json` first — the sentinel — so a checkpoint being
rewritten is visibly uncommitted. Each payload file is then written through
`writeFileAtomic` (temp file + atomic rename, so no reader ever sees a
partial file). `saveTrainerState` writes the sentinel LAST, itself
atomically. Consequence: a directory with a parseable `trainer_state.json`
is a complete, committed checkpoint; a crash mid-save leaves a sentinel-less
directory that resume logic must treat as absent (the previous sentinel was
deleted up front, so a torn save can never masquerade as committed). All optional state fields
serialize only when set and parse to `null` when absent (older checkpoints
stay readable); `format` must be `"fucina.training_checkpoint"` and
`version` 1 (`UnsupportedTrainerStateVersion` otherwise). With gradient
accumulation, save only at window boundaries — accumulated gradients live
only in GradStates and are never serialized ([TRAINING.md](../TRAINING.md)
[§4](04-tensor-operations.md)). The es_* fields make an ES checkpoint self-describing without any
`optimizer.fucina` ([§11.11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#1111-evolution-strategies-srceszig)). `examples/spirals/main.zig` `saveCheckpoint` /
`loadCheckpoint` is the reference composition of these helpers with
`ParamRegistry` and `saveState`.

## 11.10 LoRA adapters (`src/lora.zig`)

For a frozen linear weight `W: [out, in]` — f32, f16, bf16, or a
block-quantized constant, anything `dot` accepts as a frozen RHS ([§5](05-automatic-differentiation.md) routes
gradients to the f32 LHS only; constants carry no GradState) — an adapter
learns the additive update

```text
y = base(x) + (alpha / r) * dropout(x) · A^T · B^T
```

with `A: [r, in]` kaiming-uniform (seeded, deterministic — the PyTorch
`nn.Linear`/LoRA-A init) and `B: [out, r]` zeros, so the initial delta is
exactly zero. Only A and B train; the base never changes.

```zig
pub const rank_tag: Tag = .lora_r;                  // reserved rank-axis tag
pub const LoraError = error{ InvalidRank, InvalidDropout };
pub const Config = struct { rank: usize, alpha: f32, dropout_p: f32 = 0 };

pub fn Adapter(comptime in_tag: Tag, comptime out_tag: Tag) type {
    // in_tag != out_tag; neither may be .lora_r (compile errors)
    pub const ATensor = Tensor(.{ rank_tag, in_tag });
    pub const BTensor = Tensor(.{ out_tag, rank_tag });
    pub const Config = ...;                          // re-export
    a: ATensor, b: BTensor, scale: f32, dropout_p: f32,

    pub fn init(ctx: *ExecContext, in_dim: usize, out_dim: usize, config: Config, seed: u64) !Self
    pub fn deinit(self: *Self) void
    pub fn Delta(comptime XPtr: type) type           // x's tags with in_tag -> out_tag
    pub fn delta(self: *const Self, ctx: *ExecContext, x: anytype, dropout_seed: ?u64) !Delta(@TypeOf(x))
    pub fn apply(self: *const Self, ctx: *ExecContext, x: anytype, base: anytype, dropout_seed: ?u64) !Delta(@TypeOf(x))
    pub fn registerParams(self: *Self, opt: anytype, comptime name_prefix: []const u8) !void
    pub fn namedTensors(self: *const Self, comptime name_prefix: []const u8) ![2]optim.NamedTensor
    pub fn namedTensorsMut(self: *Self, comptime name_prefix: []const u8) ![2]optim.NamedTensorMut
    pub fn mergeInto(self: *const Self, ctx: *ExecContext, w: anytype) !void
    pub fn mergeF16(self: *const Self, ctx: *ExecContext, w: anytype) !W // W = w's f16 tensor type; NEW tensor
}
```

- `init` validates `1 <= rank <= min(in_dim, out_dim)`
  (`LoraError.InvalidRank`) and `0 <= dropout_p < 1`
  (`LoraError.InvalidDropout`). `seed` drives A's fill deterministically
  (same seed → bitwise-identical A). A and B are caller-owned VARIABLES —
  never scope-adopted; keep them alive as long as any optimizer or
  state-dict entry borrows them; pair with `deinit`. The effective scaling
  `alpha/rank` makes `alpha` transfer across ranks.
- `delta` computes `scale * dropout(x)·A^T·B^T`; `apply` adds a
  caller-supplied `base` (an f32 facade tensor carrying exactly the delta's
  tags — e.g. the frozen-path output `x.dot(ctx, &w, in_tag)`).
  `dropout_seed` selects the mode: a fresh per-step seed trains (consumed
  only when `dropout_p > 0`; derive per step/layer with `rng.at` — reusing
  a seed reuses the mask), `null` is eval and skips dropout entirely
  (identical to the `p == 0` zero-copy identity path). Input validation is
  comptime: x must be f32, carry `in_tag`, and carry neither `out_tag` nor
  `.lora_r`.
- **Composite-op contract**: `delta`/`apply` build a multi-op chain and
  release interior tensors on return, so any call whose result will be
  `backward()`'d MUST run under an exec scope ([§6.3](06-the-execution-runtime-execcontext-and-the-memory-model.md#63-exec-scopes-implicit-ownership-for-training-srcexeczig-srcexecruntimezig)) — without one the
  released interior graph nodes dangle (UB). Without backward (eval), no
  scope is needed.
- `registerParams(opt, "prefix")` registers A/B via `addParamNamed` as
  `"prefix.lora_a"` / `"prefix.lora_b"`; `namedTensors`/`namedTensorsMut`
  produce the matching state-dict entries. Adapter names double as the
  on-disk schema ([§11.6](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#116-paramregistry-srcparam_registryzig)).
- `mergeInto(ctx, &w)` folds the adapter into an f32 base IN PLACE
  (`w += scale·B·A`; `w: [out_tag, in_tag]`, dims checked at runtime →
  `error.ShapeMismatch`). It goes through the facade's `data()` gate, which
  only grants mutable access to no-grad tensors — a variable base returns
  `error.MutableDataRequiresNoGrad`, exactly the right fence: only frozen
  bases merge. `mergeF16` widens an f16 base to f32, merges, casts back,
  and returns a NEW f16 tensor (caller-owned). Quantized bases are NOT
  mergeable in place, deliberately: dequantize→merge→re-encode compounds
  quantization error. In memory, dequantize to f32 (`.to(ctx, .f32)`) and
  merge into the copy; for files, merge into a dense f32/f16/bf16 base and
  quantize the RESULT (below).

```zig
test "LoRA: zero delta at init; eval forward; f32 merge parity" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var adapter = try fucina.lora.Adapter(.in, .out).init(&ctx, 8, 4, .{ .rank = 2, .alpha = 4 }, 42);
    defer adapter.deinit();

    var x_vals: [16]f32 = undefined;
    fucina.rng.uniformFill(1, &x_vals, -1, 1);
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 2, 8 }, &x_vals);
    defer x.deinit();
    var w_vals: [32]f32 = undefined;
    fucina.rng.uniformFill(2, &w_vals, -1, 1);
    var w = try fucina.Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 4, 8 }, &w_vals);
    defer w.deinit();

    var base = try x.dot(&ctx, &w, .in); // frozen-base forward
    defer base.deinit();
    var y0 = try adapter.apply(&ctx, &x, &base, null); // null seed = eval
    defer y0.deinit();
    // B is zero-initialized: apply returns the base bitwise.
    try std.testing.expectEqualSlices(f32, try base.dataConst(), try y0.dataConst());

    fucina.rng.uniformFill(3, adapter.b.value.data(), -0.5, 0.5); // stand-in for training
    var y1 = try adapter.apply(&ctx, &x, &base, null);
    defer y1.deinit();

    try adapter.mergeInto(&ctx, &w); // w += (alpha/rank) * B*A, in place
    var y2 = try x.dot(&ctx, &w, .in); // the merged weight alone
    defer y2.deinit();
    for (try y1.dataConst(), try y2.dataConst()) |expected, got| {
        try std.testing.expectApproxEqAbs(expected, got, 1e-4); // fp order differs, not bitwise
    }
}
```

**Fine-tune → merge → quantize → serve.** LLM-scale LoRA fine-tuning lives
in `models.qwen3.train` ([§13](13-the-model-stack-fucina_models.md)): `Trainer(targets)` puts adapters on selected
projections of a frozen GGUF model, `saveAdapters`/`loadAdapters` persist
them as `adapters.safetensors` (names `layers.<i>.<target>.lora_{a,b}`;
`loadAdaptersWithOptions` threads `state_dict.LoadOptions`). The loop back
to a servable model is `zig build export-gguf` (merge and quantize are
separate passes BY DESIGN — one combined pass would chain-requantize):

```sh
zig build finetune -Doptimize=ReleaseFast -- \
    --model models/Qwen3-0.6B-f16.gguf --steps 30 --save /tmp/qwen3-lora
zig build export-gguf -Doptimize=ReleaseFast -- \
    --from-gguf models/Qwen3-0.6B-f16.gguf --adapters /tmp/qwen3-lora \
    --alpha 16 --out /tmp/qwen3-tuned-f16.gguf          # merge (dense f32/f16/bf16 base only)
zig build export-gguf -Doptimize=ReleaseFast -- \
    --from-gguf /tmp/qwen3-tuned-f16.gguf --dtype q4_k --out /tmp/qwen3-tuned-q4_k.gguf
zig build qwen3 -Doptimize=ReleaseFast -- /tmp/qwen3-tuned-q4_k.gguf --chat "..."
# or: llama-cli -m /tmp/qwen3-tuned-q4_k.gguf            # any GGUF consumer
```

The adapter checkpoint stores A/B but not alpha — pass the training-time
value to `--alpha`. Details, transcode policy, and gradient-verification
evidence: [TRAINING.md](../TRAINING.md) [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md).

## 11.11 Evolution strategies (`src/es.zig`)

`fucina.es` trains WITHOUT gradients — a faithful reimplementation of
ES-at-scale (arXiv:2509.24372; algorithm reimplemented from the paper, the
reference code being under a noncommercial license): deliberately vanilla
OpenAI-ES with the reference's simplifications kept intact —

```text
eps_n ~ N(0, I)                               n = 1..population
R_n   = reward(theta + sigma * eps_n)         (forward passes only)
C_n   = (R_n - mean(R)) / (std(R) + 1e-8)     (z-score, f64 stats, ddof = 0)
theta += (alpha / population) * sum_n C_n * eps_n
```

No antithetic pairs, no rank shaping, no optimizer state, and no 1/sigma in
the update (folded into alpha; the reference default is `alpha = sigma/2`).
Because the signal is one scalar reward per member, ES composes with
anything scoreable from a forward pass, and every parameter is fair game —
no `GradState` needed: f32 variables, f32 constants, typed f16/bf16
tensors, whole registries, and packed ternary genomes all register.

```zig
pub const EsError = error{ InvalidConfig, AnchorMissing, UnsupportedDType,
    NonContiguousParam, DuplicateParam, NoParams, RewardCountMismatch,
    MemberActive, MemberNotActive, ReplicaShapeMismatch };
pub const NoiseScheme = enum { iid, correlated };
pub const RestoreMode = enum { regenerate, snapshot };
pub const AnchorDecay = enum { none, l1, l2 };
pub const RewardNorm  = enum { z_score, centered_ranks, none };
pub const Stats = struct { mean_reward: f64, std_reward: f64, min_reward: f32, max_reward: f32 };
pub const BlockTQ2_0 = ...;                    // re-export (ternary genomes, §10)

pub const Config = struct {
    sigma: f32 = 0.001,
    alpha: ?f32 = null,                        // null = sigma/2
    population: usize = 30,
    antithetic: bool = false,                  // mirrored (+eps, -eps) pairs; even population
    noise: NoiseScheme = .iid,
    restore_mode: RestoreMode = .regenerate,
    cache_streams: bool = false,
    anchor_decay: AnchorDecay = .none, anchor_lambda: f32 = 0,
    reward_norm: RewardNorm = .z_score,
    ternary_flip_rate: f32 = 0.001, ternary_update_fraction: f32 = 0.005,
    ternary_update_decay: f32 = 0.0,
    seed: u64 = 42,
};
```

```zig
pub const Trainer = struct {
    iteration: u64 = 0,                        // advances once per update; persist to resume
    pub fn init(allocator: Allocator, config: Config) !Trainer
    pub fn deinit(self: *Self) void
    pub fn applyResumedConfig(self: *Self, config: Config) !void
    pub fn alphaValue(self: *const Self) f32
    pub fn addParam(self: *Self, t: anytype) !void
    pub fn addParamNamed(self: *Self, t: anytype, name: ?[]const u8) !void
    pub fn addRegistry(self: *Self, registry: *const ParamRegistry) !usize
    pub fn addTernaryParam(self: *Self, blocks: []BlockTQ2_0, len: usize) !void
    pub fn addTernaryParamNamed(self: *Self, blocks: []BlockTQ2_0, len: usize, name: ?[]const u8) !void
    pub fn captureAnchor(self: *Self) !void
    pub fn paramCount(self: *const Self) usize
    pub fn elementCount(self: *const Self) usize
    pub fn memberSeed(self: *const Self, member: usize) u64
    pub fn ternaryMemberSeed(self: *const Self, member: usize) u64
    pub fn ternarySlotStreamSeed(member_seed: u64, slot_index: usize) u64
    pub fn ternaryFlipCount(self: *const Self, len: usize) usize
    pub fn perturb(self: *Self, ctx: *ExecContext, member: usize) !void
    pub fn restore(self: *Self, ctx: *ExecContext, member: usize) !void
    pub fn materializeMember(self: *const Self, member: usize, dst: []const []u8) !void
    pub fn materializeTernaryMember(self: *const Self, member: usize, dst: []const []BlockTQ2_0) !void
    pub fn update(self: *Self, ctx: *ExecContext, rewards: []const f32) !Stats
    pub fn step(self: *Self, ctx: *ExecContext, evaluator: anytype) !Stats
    pub fn evaluateMembers(self: *const Self, evaluator: anytype, rewards: []f32, workers: usize) !void
};
```

**Registration.** `addParam`/`addParamNamed` accept a pointer to any
f32/f16/bf16 facade tensor — autograd variable, constant, or grad-free
typed tensor; ES treats them all the same (compile error otherwise;
`NonContiguousParam` at runtime). The trainer retains a refcounted storage
view (the facade may move by value); names are borrowed. `addRegistry`
registers every entry of a `ParamRegistry` — trainable AND frozen —
BORROWING buffers and names (the registry must outlive the trainer);
entries whose storage is already registered are SKIPPED, not rejected
(tied weights perturb once, matching torch `named_parameters()`
deduplication), and the added-slot count is returned. Duplicate storage via
`addParam` is `DuplicateParam`; adding any slot while a member is applied
in place is `MemberActive`. `init` validates the config
(`InvalidConfig`): sigma/alpha positive-finite, population ≥ 2, even under
`antithetic`, ternary knobs in range, and for AWD a positive-finite lambda
with `alpha*lambda < 1` for l2. `config` is readable after `init`, but a
resume replaces it through `applyResumedConfig(config)`, never by field
assignment: the same validation runs on the whole new config (an odd
population under `antithetic` is rejected on resume exactly as at `init`,
leaving the live config untouched), every ternary slot's undo log is
re-sized to the new per-member flip count, and the per-iteration noise
cache is dropped so no stream filled under the old seed or scheme
survives. Rejected with `MemberActive` while a member is applied in place.

**Seed-regenerated noise (the scale trick).** Noise is never stored: a
member's perturbation is a pure function of
`(config.seed, iteration, member, slot, element)` through the counter-based
gaussian `rng.gaussianFillAtFast` (vectorized; a distinct checkpoint
contract from the scalar `gaussianFillAt` mapping that APOLLO stays on), so
`perturb`, `restore`, and `update` regenerate it on the fly — O(1) memory
beyond the parameters. `memberSeed(member)` exposes the derivation
(domain-separated `rng.at`); the mapping may never change once checkpoints
exist. `NoiseScheme.iid` (default) gives every (member, slot) an
independent stream; `.correlated` reuses ONE stream per member across slots
— same-length slots get identical noise — mirroring the reference library's
acknowledged reseeding artifact (kept for reference-faithful runs). Both
are checkpoint contracts: `(config.seed, iteration)` plus population and
the scheme knobs fully regenerate the population, so resume needs only the
iteration counter (`TrainerState.es_*` fields — there is no
`optimizer.fucina` in an ES checkpoint; on resume, hand the saved
sigma/alpha/population/scheme/seed to `applyResumedConfig` and restore
`es_iteration`).

**Two evaluation shapes.**

- *In place* (big models): `perturb(ctx, member)` mutates the registered
  buffers (`theta += sigma*eps`, chunk-parallel); exactly one member may be
  active (`MemberActive` tripwire; `restore` of the wrong member is
  `MemberNotActive`). `restore` undoes it: `.regenerate` (default)
  subtracts the regenerated noise — exact up to `(x+t)-t` float drift — or
  `.snapshot` memcpys parameter bytes back (bitwise, costs one parameter
  copy). `step(ctx, evaluator)` is the sequential driver:
  perturb → `evaluator.eval(member) !f32` → restore per member, then one
  `update`; on an eval error it restores before propagating.
- *Member-parallel replicas* (small parameter sets): `materializeMember`
  writes `theta + sigma*eps_member` into caller-owned replica buffers
  (`dst[k]` = slot k in registration order, byte lengths validated —
  `ReplicaShapeMismatch`; buffers must be scalar-aligned) without touching
  shared theta; `evaluateMembers(evaluator, rewards, workers)` fans members
  out over OS threads pulling from a shared counter, calling
  `evaluator.evalMember(worker, member) !f32` — `worker` indexes the
  caller's replica table. `workers` ≥ 1 (clamped to population and 64); the
  evaluator must be thread-safe across distinct workers; the first error
  stops the fan-out and is returned after all workers join.
  `examples/es_spirals/main.zig` is this shape end-to-end (each worker owns a
  replica model + its own ExecContext; only scalar rewards cross threads).

**The update.** `update(ctx, rewards)` (`rewards.len == population`, else
`RewardCountMismatch`; `NoParams` with nothing registered) computes reward
stats in f64 (ddof 0, sequential summation), shapes coefficients per
`reward_norm` — `.z_score` (affine: preserves outlier magnitude, so one
catastrophic member can dominate with unbounded rewards; self-stops exactly
on all-equal rewards), `.centered_ranks` (Salimans centered ranks in
[-0.5, 0.5], monotone-invariant, outlier-immune; ties break by member
index, a pinned total order; all-equal rewards still take a mean-zero
phantom step), `.none` (raw coefficients) — then applies
`theta += (alpha/population)·Σ C_n·eps_n` chunk-parallel with fp32
accumulation and pinned rounding placement (mul-then-add, no FMA; member
order fixed inside each element), narrowing once to the parameter dtype.
Under `antithetic`, pairs fold to `(C_2k - C_2k+1)·eps_k`, halving
update-side regeneration; division stays by the full population. `update`
is mutate-last: every fallible step runs before the first parameter byte
changes, so a failed update is a no-op. It returns the pre-normalization
`Stats` and advances `iteration`. Determinism: all kernels are
element-independent maps, bitwise identical to the serial loop for ANY
thread count; `cache_streams = true` regenerates each stream once per
iteration into a RAM cache and replays it, bitwise-neutral (worthwhile only
when replays dominate — large populations, small parameter sets).

**Anchored weight decay** (arXiv:2605.30148): with
`anchor_decay = .l1|.l2` and `anchor_lambda`, each `update` ends with a
proximal pull toward a fixed anchor — l2 shrinks `theta - theta_0` by
`(1 - alpha*lambda)`, l1 soft-thresholds it at `alpha*lambda` (exact
zeroing; sparsity in the fine-tuning delta) — counteracting random-walk
drift in reward-irrelevant directions. `captureAnchor` snapshots the
CURRENT float parameters as theta_0: call once after registration while
the parameters still hold the pretrained values, in particular BEFORE
loading a checkpoint on resume (the anchor is never serialized;
`update` without it is `AnchorMissing`). Fine-tuning only — anchoring a
random init pins the model to noise. Reference values: l2 lambda 10, l1
lambda 0.01 at alpha 5e-4.

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

**Parity evidence.** Three layers: `tools/gen_es_goldens.py` replicates the
repo RNG bit-level and the update algebra in numpy (tolerance goldens in
`src/es_tests.zig` — the generator's f64 libm gaussian sits a few f32 ulps
from the `gaussianFillAtFast` polynomials); a
test-local straight-line serial reference pins the chunk-parallel kernels
BITWISE; and `tools/check_es_parity.py` runs the ACTUAL reference code
(es-at-scale perturb/restore/z-score/update, and es-awd's decay kernels)
against torch transcriptions of es.zig's algebra on identical noise —
bitwise `torch.equal` on f32/f16/bf16, both noise schemes. The one
deliberate substitution is the RNG itself (repo-owned splitmix64 instead of
torch Philox — a checkpoint contract). On `-Dgpu=cuda` builds, GPU-resident
slots run perturb/restore/update/anchor as device kernels bitwise-identical
to the CPU path for any launch geometry, so checkpoints are
device-independent; non-resident slots, bf16, and active stream caches fall
back per slot ([TRAINING.md](../TRAINING.md) [§13](13-the-model-stack-fucina_models.md)).

**Practical notes.** Rewards must be finite — one NaN/Inf poisons the
z-score for the whole iteration (clamp in the evaluator; the reference
scores failed rollouts 0.0). Prefer BOUNDED rewards, or
`reward_norm = .centered_ranks` for unbounded ones (raw −CE) — neither
normalization shrinks the step near an optimum, so the practical brakes are
saturating rewards, conservative sigma, AWD, and eval-selected checkpoints.
Changing `population`, `seed`, `noise`, or `antithetic` mid-run breaks the
noise contract exactly like editing an optimizer checkpoint.
`zig build es-spirals` (from-scratch two-spirals training, no backward
anywhere; self-verifying against `--target`, default 0.90 accuracy — 100 %
is the typically observed result) and `zig build es-finetune` (finetune.zig's
gradient-free twin: `--mode lora` perturbs adapters, `--mode full` every
resident float weight — quantized blocks cannot take noise; rewards
`rule`/`acc`/`nll`) are the reference applications
([TRAINING.md](../TRAINING.md) [§13](13-the-model-stack-fucina_models.md)).

**Ternary-native ES.** `addTernaryParam`/`addTernaryParamNamed` register
BORROWED packed TQ2_0 genomes (`len` a positive multiple of 256,
`blocks.len == len/256`, every 2-bit crumb a valid ternary code — corrupt
code 3 is rejected at registration): the packed blocks ARE the training
state, so every member and every checkpoint is a servable ternary model
(training = packed inference model). Perturbation is sparse trit flips from
the dedicated `es_trits` counter-RNG domain (`ternaryMemberSeed` /
`ternarySlotStreamSeed` / `ternaryFlipCount` expose the pinned mappings;
float slots stay bitwise unchanged when ternary slots join); restore
replays a per-slot undo log in reverse (clamping at the rails is lossy, so
regenerate-subtract cannot work); the update is EGGROLL-style
vote-and-threshold top-K one-bin moves; `materializeTernaryMember` is the
replica twin. Block scales (`d`) are never touched; ternary slots skip
snapshots, stream caches, and AWD. The three `ternary_*` config knobs are
checkpoint contracts like `antithetic`: (seed, iteration, population, these
rates) regenerate every member's flips and the update schedule, so a
resumed run must re-pass them identically (the trainers' `TrainerState`
does not persist them). Quantization
background and the TQ2_0 layout: [§10](10-quantization.md); design record:
[TERNARY.md](../TERNARY.md); acceptance demo: `zig build es-ternary-spirals`.
