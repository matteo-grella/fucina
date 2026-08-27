# 5. Automatic differentiation

Fucina's autograd is eager and backward-only: every op computes its value
immediately through the same public code path, and — when an operand requires
gradients — attaches a backward record for the reverse pass. There is no
graph compiler, no tape replay, and no separate "raw" autograd surface (a
guard test in `src/ag_tests.zig` asserts the legacy `Function`/`Node`/`Engine`
declarations stay removed from the core). The module root is `src/ag.zig`;
the user-facing surface in this section (`Tensor`, `checkpoint`,
`checkpointWithContext`, `noGrad`, `isGradEnabled`, `NoGradScope`,
`customVjp`, `gradcheck` and its option/result types) is re-exported at the
`fucina` root ([§1](01-introduction-and-mental-model.md)). The engine internals documented below — `GradState`,
`BackwardFunction`, `AgError`, the `backwardGrad*` entry points,
`BlockOutput`/`BlockOutputWithContext` — deliberately are not.

## 5.1 The gradient model (`src/ag/tensor.zig`, `src/ag/core.zig`)

The f32 public tensor ([§3](03-tensors-types-construction-and-data-access.md)) owns exactly one raw value and at most one
gradient state:

```zig
value: RawTensor,
grad_state: ?*GradState = null,
scope_owned: bool = false, // exec-scope borrow flag, see §6
```

- **Constants** (`constant`, `fromSlice`, `fromTensor`, `zeros`, `ones`,
  `full`, `scalar`, `empty`, the borrowed-slice constructors) have
  `grad_state == null`. They participate in any op but never accumulate
  gradients.
- **Variables** (`variable`, `variableFromSlice`) attach a leaf `GradState`
  allocated from `ctx.allocator()`. `deinit` on the tensor releases both the
  value and the state (unless `scope_owned`, in which case `deinit` is a
  no-op and the exec scope releases everything at `closeExecScope`, [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)).
- The f32 branch carries the full graph machinery. The f16/bf16 branch is
  a LEAF-capable participant: `variable`/`variableFromSlice` create
  trainable 16-bit leaves whose accumulated gradient is ALWAYS an f32 raw
  tensor (there is no 16-bit gradient anywhere in the engine), and the
  differentiable entries into/out of the 16-bit world are `to` (both cast
  directions, [§3.8](03-tensors-types-construction-and-data-access.md#38-casting-todtype-srcagtensorzig-srcexecconvertzig)) and the mixed-RHS `dot`/`einsum` ([§4.8](04-tensor-operations.md#48-dot-tag-directed-contraction-srcagtensorzig-srctag_opszig)). Every OTHER
  typed forward op is no-grad by design and rejects a grad-requiring
  operand with `error.UnsupportedGradient` ([§4.19](04-tensor-operations.md#419-math-on-non-f32-tensors-srcagtensorzig)). f64, integer, and
  block-quantized constant branches have no gradient support; their
  `requiresGrad()` never returns `true` (integer and block-quantized
  branches hard-wire `false`; on f64 `variable`/`variableFromSlice`/
  `grad`/`gradView` exist but are compile errors, while `zeroGrad` no-ops
  and `detach` just returns a no-grad view).

`requiresGrad` is simply:

```zig
pub fn requiresGrad(self: *const Self) bool // grad_state != null
```

Every differentiable op funnels through one private tail (`finishOp`): if no
operand requires gradients — or a `noGrad` scope is active ([§5.4](05-automatic-differentiation.md#54-nograd-scopes-srcagcontrolzig)) — the
result is a plain no-grad tensor and no graph state is retained; otherwise
the eager value is wrapped together with a VJP record from
`src/ag/backward/` inside a fresh `GradState`. Because forward always
takes the identical kernel path, training and inference produce identical
values.

A `GradState` (`src/ag/core.zig`) holds the accumulated gradient, the
backward record, and the scheduling state:

```zig
pub const GradState = struct {
    allocator: Allocator,
    grad: ?Tensor = null,              // raw accumulated gradient
    grad_fn: ?BackwardFunction = null, // null for leaves
    state: std.atomic.Value(u8),       // idle | pending | ongoing
    pending_grads: std.atomic.Value(u32),
    grad_mutex: thread.Mutex,
    backward_done: bool,               // completed pass consumed this output (see 5.2)
    pass_output: bool,                 // outputs keep their gradient; interiors release on consume (see 5.2)
    refs: std.atomic.Value(u32),       // one per handle / consumer-record operand / scope entry
};
```

`BackwardFunction` is the type-erased VJP record interface:

```zig
pub const BackwardFunction = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        operands: *const fn (*const anyopaque) []const ?*GradState,
        backward: *const fn (*anyopaque, *ExecContext, *const Tensor,
                             []?Tensor) anyerror!void,
        deinit: *const fn (*anyopaque, Allocator) void,
        prefer_async_backward: bool = false,
        estimated_work: ?*const fn (*const anyopaque) usize = null,
    };
};
```

`operands()` returns one slot per forward operand (`null` for operands that
were constants); `backward(ctx, gy, out)` must write an owned raw gradient
into `out[i]` for every operand slot that holds a state (`core.needs`,
[§5.2](05-automatic-differentiation.md#52-running-backward-srcagcorezig)). The engine consumes and deinits those tensors. This interface is internal; user-defined
differentiable ops go through `customVjp` ([§5.6](05-automatic-differentiation.md#56-custom-vjps-srcagcustomzig)), which implements it for
you.

```zig
test "backward and grad read" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var c = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer c.deinit(); // constant: no grad state

    var y = try x.mul(&ctx, &c);
    defer y.deinit();
    var loss = try y.sumAll(&ctx); // scalar output: implicit seed 1
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?; // deep copy; caller deinits
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30 }, try gx.dataConst());
    try std.testing.expect((try c.grad(&ctx)) == null); // constants never accumulate
}
```

## 5.2 Running backward (`src/ag/core.zig`)

The facade exposes a single-output entry point, in an implicitly-seeded and
an explicitly-seeded form:

```zig
pub fn backward(self: *Self, ctx: *ExecContext) !void
pub fn backwardWithGrad(self: *Self, ctx: *ExecContext, grad_output: *const Self) !void
```

Both error with `error.NoGradientGraph` when called on a tensor without a
`grad_state` and otherwise delegate to the engine's
`core.backwardGradOne(ctx, state, &self.value)`. `backwardWithGrad` first
installs `grad_output` as the output gradient: it is same-tagged (checked
at comptime) and must match `self`'s shape (`error.ShapeMismatch`); it is
read as a *value* — its own gradient state, if any, is ignored — and
replaces any gradient already held by `self`. The engine itself is
multi-output-capable but internal (not re-exported at the `fucina` root):

```zig
pub fn backwardGrad(ctx: *ExecContext, outputs: []const *GradState,
                    output_values: []const *const Tensor) !void
pub fn backwardGradSerial(ctx: *ExecContext, ...) !void // same, node-serial
pub fn backwardGradOne(ctx: *ExecContext, output: *GradState,
                       output_value: *const Tensor) !void
```

`pub const AgError = error{ MissingOutputGradient, MissingBackwardGradient, BackwardAlreadyRun };`

**Seeding rules.** Before any scheduling state exists, every output is
validated and its implicit seed pre-allocated:

- A **scalar** output (one element total — a `{1,1}` tensor counts) with no
  gradient present receives the implicit seed `1`.
- A **non-scalar** output with no gradient present fails with
  `error.MissingOutputGradient` — seed it with `backwardWithGrad` (or
  install a gradient through the low-level `setGrad`, [§5.3](05-automatic-differentiation.md#53-reading-seeding-and-resetting-gradients-srcagtensorzig-srcagcorezig)).
- An output whose `GradState` **already holds a gradient** (installed by
  `backwardWithGrad`, by `setGrad` ([§5.3](05-automatic-differentiation.md#53-reading-seeding-and-resetting-gradients-srcagtensorzig-srcagcorezig)), or by the checkpoint recompute)
  is respected as-is — the implicit `+1` is *not* added on top.
- In a multi-output pass, a scalar output whose gradient appears only
  **mid-pass** (an earlier output's backward already contributed to it)
  still accumulates its own seed on top of that contribution.

```zig
test "non-scalar output needs a seed" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer x.deinit();
    var y = try x.scale(&ctx, 2);
    defer y.deinit();

    // Unseeded non-scalar output: fails before any scheduling state exists,
    // so the SAME graph runs once a seed is supplied.
    try std.testing.expectError(error.MissingOutputGradient, y.backward(&ctx));

    var grad_output = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 1, 10 });
    defer grad_output.deinit();
    try y.backwardWithGrad(&ctx, &grad_output); // shape-checked; read as a value
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 20 }, try gx.dataConst());
}
```

**Pending-counter scheduling.** `prepareBackwardPass` walks the graph once,
incrementing each state's `pending_grads` by one per consumer edge and
flipping it `idle → pending` on first visit. A node's VJP executes only when
its counter drains to zero — i.e. after *all* downstream contributions have
arrived — so a value consumed by several branches (a shared activation, a
weight used twice) accumulates the complete upstream gradient before
propagating it exactly once.

```zig
test "shared branch accumulates" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 3, 5 });
    defer x.deinit();

    var sq = try x.mul(&ctx, &x); // x feeds the node twice
    defer sq.deinit();
    var lin = try x.scale(&ctx, 4); // second consumer of x
    defer lin.deinit();
    var both = try sq.add(&ctx, &lin);
    defer both.deinit();
    var loss = try both.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    // d/dx (x^2 + 4x) = 2x + 4: contributions from every branch summed.
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 14 }, try gx.dataConst());
}
```

**Accumulation mechanics.** Contributions add in place under the per-state
`grad_mutex`, so concurrent branches on pool threads are safe. Before the
engine mutates an accumulator (or installs a first contribution that will be
added to later), it checks the raw tensor's exclusive-ownership predicate and
materializes a private copy when the buffer is shared — a VJP may therefore
hand back cheap refcounted *views* of `gy` without risking cross-state
aliasing (`src/ag/core_tests.zig` pins this copy-on-write behavior).

**Operand pruning.** A VJP fills `out[i]` only for the operand slots that
hold a `GradState` (`core.needs(record, i)`), so gradients for constant
operands — frozen weights, masks, cached KV — are never computed. The
`customVjp` Spec keeps its `needs_grad: []const bool` view of the same
information.

**Parallel backward vs `backwardGradSerial`.** The engine grabs the
context's work pool (`ctx.tryWorkPool()`, [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)). When several *independent*
states become ready at once, it may spawn all but one onto pool threads —
but only for records that opt in through the vtable: `prefer_async_backward
= true` (no in-tree record currently sets it) or an `estimated_work()` at or
above `parallel.backward_async_work_threshold` (`256 * 1024 * 1024` work
units; provided by the attention, causal-conv1d-family, gather,
linear-cross-entropy, linear-distill, `Conv1d`/`Conv2d`, `Dot`, `AddDot`,
and ternary-STE-dot records). Node-level spawning is
additionally gated at comptime by `exec.parallel_dot_backward_branches`
(native backend with BLAS, [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)) — on scalar or no-BLAS builds every node runs
inline on the calling thread. (`DotBackward` and `AddDotBackward`
additionally parallelize their two contraction branches internally, via
the context's dot-backward worker.)
`backwardGradSerial` forces `pool = null` so the whole pass is node-serial
regardless; kernel-level `parallelChunks` parallelism *inside* a VJP is
unaffected. Serial mode is required whenever a threadlocal guard must
observe the entire pass on one thread — the checkpoint recompute ([§5.5](05-automatic-differentiation.md#55-activation-checkpointing-srcagcheckpointzig)) is
the in-tree case.

**Error exits are re-runnable.** Seeds are validated and allocated *before*
any pending counter is installed — a stranded counter would make the next
backward over the same states stop at their `.pending` check and report
success with missing gradients — so a seeding failure (as in the snippet
above) leaves zero scheduling debris and the same graph runs correctly once
the seed is supplied (`src/ag/core_tests.zig`, "failed output seeding leaves
the graph re-runnable"). When a VJP fails mid-pass, the engine deinits any
gradients it produced, releases the pending counters of the failing node's
operands (returning them to `idle`), records the first error, drains all
in-flight tasks, and returns that error. Re-runnability restores
*scheduling* state, not values: gradient contributions delivered before the
failure remain accumulated — call `zeroGrad` on the leaves before retrying
if exact values matter.

**Interior gradients live only as long as they are needed.** Gradients
accumulate in every `GradState` they touch, but an interior result's
gradient is released as soon as that result's own backward has consumed
it: the backward pass holds a moving window of gradients, not a second copy
of the forward (on the Qwen3-0.6B LoRA step at 1280 tokens this is the
difference between a 23.6 GB and a 15.6 GB peak). What a completed pass
leaves behind are the gradients of its leaves (for the optimizer) and of
the outputs it was asked for (`backward` over several outputs keeps each
output's gradient readable, including an output that is interior to
another). To read any other interior gradient, pass that tensor as an
additional output.

**One backward per graph.** A completed pass marks its outputs consumed,
and a repeated `backward`/`backwardWithGrad` over them fails with
`error.BackwardAlreadyRun` before any scheduling state is installed;
`zeroGrad` resets gradients, not the consumed graph. Only a *completed*
pass consumes: the failed-seeding retry above stays re-runnable, and a leaf
output (a bare variable) has no graph to consume and is never marked. The
supported accumulation idiom is one backward per freshly built forward
graph over shared leaves ([§5.3](05-automatic-differentiation.md#53-reading-seeding-and-resetting-gradients-srcagtensorzig-srcagcorezig)).

## 5.3 Reading, seeding, and resetting gradients (`src/ag/tensor.zig`, `src/ag/core.zig`)

```zig
pub fn grad(self: *const Self, ctx: *ExecContext) !?Self     // deep copy
pub fn gradView(self: *const Self, ctx: *ExecContext) !?Self // refcounted view
pub fn zeroGrad(self: *Self) void                            // drop accumulated grad
pub fn detach(self: *const Self, ctx: *ExecContext) !Self    // no-grad view of value
```

- `grad` returns `null` for constants and for variables with no accumulated
  gradient; otherwise a caller-owned no-grad tensor holding a deep copy
  (allocated from `ctx.allocator()`). `gradView` is the zero-copy variant: the
  result aliases the accumulator *as of that moment*. A later backward pass
  does **not** mutate it — the held reference defeats the engine's
  copy-on-write check (`canTakeInPlace` needs a unique buffer), so the
  engine accumulates into a fresh private buffer and the view silently keeps
  the stale pre-pass value. Use `gradView` for immediate reads, `grad` for
  anything that must observe later passes. Both are taken under the state's
  mutex.
- `zeroGrad` frees the accumulated gradient (no-op on constants). Training
  loops call it between optimizer steps; `optim.OptimizerSet.zeroGrad` ([§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md))
  fans it out over registered parameters.
- `detach` returns a no-grad constant sharing the same storage
  (refcounted view): the value flows on, the graph is cut.
- `data()` refuses mutable access on a grad-carrying tensor with
  `error.MutableDataRequiresNoGrad` (mutating a recorded value would
  invalidate the graph); `dataConst()`/`item()`/`copyTo()` are always
  allowed.

Direct gradient state access goes through the public `grad_state` field
(`?*GradState`); its methods are thread-safe under the per-state mutex:

```zig
pub fn setGrad(self: *GradState, grad: Tensor) void        // takes ownership, replaces
pub fn zeroGrad(self: *GradState) void
pub fn gradClone(self: *GradState, allocator: Allocator) !?Tensor
pub fn gradView(self: *GradState) !?Tensor
```

`setGrad` consumes a *raw* tensor (e.g. produced by `ctx.fromSlice`, [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)) and
replaces any existing gradient. For seeding an output before `backward`,
prefer `backwardWithGrad` ([§5.2](05-automatic-differentiation.md#52-running-backward-srcagcorezig)) — it stays in facade tensors and
shape-checks the output gradient; `setGrad` is the unchecked low-level hook
underneath it (the checkpoint recompute seeds through it too, [§5.5](05-automatic-differentiation.md#55-activation-checkpointing-srcagcheckpointzig), and gradient
clipping rewrites accumulated gradients with it, [§11.4](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md#114-gradient-clipping-and-lr-schedules-srcoptimzig)).
`GradState.leaf`/`retain`/`release` (the reference count: one per handle,
one per consumer-record operand, one per exec-scope entry) and the
`createNode`/`BackwardNode` record co-allocation exist for internal wiring
and are managed by the facade.

**Accumulation across backward calls** — the micro-batch idiom: build a
fresh forward graph per micro-batch over the same leaf variables and call
`backward` once per graph; leaf gradients sum. (A repeat over the *same*
graph fails with `error.BackwardAlreadyRun`, [§5.2](05-automatic-differentiation.md#52-running-backward-srcagcorezig).)

```zig
test "micro-batch accumulation and zeroGrad" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var w = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer w.deinit();

    for (0..2) |_| { // one fresh graph per micro-batch
        var y = try w.scale(&ctx, 3);
        defer y.deinit();
        var loss = try y.sumAll(&ctx);
        defer loss.deinit();
        try loss.backward(&ctx);
    }
    var gw = (try w.grad(&ctx)).?; // 3 + 3: sums across the two passes
    defer gw.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 6 }, try gw.dataConst());

    w.zeroGrad(); // training loops reset between optimizer steps
    try std.testing.expect((try w.grad(&ctx)) == null);
}
```

When gradients are tracked, interior op results carry live `GradState`s that
downstream records retain (one reference per operand, taken when the record
is created and dropped when it is destroyed; `src/ag/core.zig`). Releasing a
handle before `backward` is therefore always safe: the graph keeps the node
alive until its last consumer record goes. The *composed* facade ops
(`nllLoss`, `l2Normalize`, `cosineSimilarity`, `norm`, `normAll`,
`maskedSelect`, `maskedScatter`, `slice` (more than one sliced
axis), `reshape` (multi-tag targets over a non-contiguous source; a
contiguous source is one strided-view record), `rollBy`, `shiftBy`,
`trace`, `diag`, `diagEmbed`, `constantPad2d`/`zeroPad2d`, `stack`,
`unbindInto`, `einsumMany`, `conv2dRelu` (its grad path is `conv2d` then
`relu`)) release their function-local intermediates on return and
differentiate scoped or unscoped alike (pinned by
`src/ag/tensor_tests/ownership.zig`); `select` is a single strided-view
record, not a composition.

## 5.4 noGrad scopes (`src/ag/control.zig`)

```zig
pub fn noGrad() NoGradScope
pub fn isGradEnabled() bool
pub const NoGradScope = struct {
    pub fn close(self: *NoGradScope) void
};
```

`noGrad()` increments a **threadlocal** depth counter and returns a scope
handle; `close()` decrements it (asserting the depth is nonzero) and is
idempotent — a handle closed early makes the deferred `close()` a no-op, as
in `src/ag/control.zig`'s own test. Scopes nest arbitrarily;
`isGradEnabled()` is true only at depth zero, per thread.

While disabled, every op takes the identical forward path but skips backward
record creation even when operands are variables — the standard evaluation
mode wrapper. `customVjp` honors it too. `checkpoint` does **not**: it keys
on its inputs' `requiresGrad` alone, so a checkpoint call inside a `noGrad`
scope still records its backward node (the block body itself always runs
grad-free; only the outer node is affected).

```zig
test "noGrad suppresses recording" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer x.deinit();

    var scope = fucina.noGrad();
    defer scope.close();
    try std.testing.expect(!fucina.isGradEnabled());

    var y = try x.scale(&ctx, 2); // same op path, but no backward node
    defer y.deinit();
    try std.testing.expect(!y.requiresGrad());
}
```

## 5.5 Activation checkpointing (`src/ag/checkpoint.zig`)

```zig
pub fn checkpoint(ctx: *ExecContext, comptime block: anytype, inputs: anytype)
    !BlockOutput(block, @TypeOf(inputs))
pub fn checkpointWithContext(ctx: *ExecContext, comptime block: anytype,
    extra: anytype, inputs: anytype)
    !BlockOutputWithContext(block, @TypeOf(extra), @TypeOf(inputs))
```

Recompute-in-backward: the forward run executes `block` on grad-free
constants inside an inner exec scope and retains only refcounted views of
the block **inputs** plus one deep copy of the block **output** — every
intermediate is freed the moment the scope closes. When gradients reach the
checkpoint node during backward, the block is re-run on the stored input
views to rebuild the subgraph, the incoming gradient is installed on the
recomputed output with `setGrad` (which is why pre-seeded outputs are never
topped up with `+1`, [§5.2](05-automatic-differentiation.md#52-running-backward-srcagcorezig)), a full inner backward runs, and the resulting
input gradients are handed to the outer engine. Memory per checkpoint is
O(inputs + output) instead of O(intermediates).

`BlockOutput`/`BlockOutputWithContext` (pub in `src/ag/checkpoint.zig`, not
re-exported at the root) compute the result type: the block's return type
with the error union stripped.

Contract for `block` (violations are compile errors where detectable):

- a comptime function `fn (*ExecContext, ...inputs) !Tensor(..)` — for
  `checkpointWithContext`, `fn (*ExecContext, extra, ...inputs) !Tensor(..)`
  — whose parameters after the lead are single-item pointers to **f32**
  facade tensors matching the `inputs` tuple, and whose result is produced
  by facade ops on those inputs. Alternatively the block may take ONE
  trailing tuple parameter carrying all inputs
  (`fn (*ExecContext[, extra], inputs: InputsTuple) !Tensor(..)`, where
  `InputsTuple` is a tuple of the same facade-pointer types) — the tuple
  form serves blocks whose input arity is comptime-variable, e.g. a LoRA
  layer with N enabled adapters; detection is by parameter type (a tuple is
  never a facade pointer), and both forms are bitwise-identical (pinned by
  tests). The block always runs under an exec scope, so the defer-deinit
  forward idiom works unchanged inside it (deinits of scope-owned results
  are no-ops).
- **deterministic and pure in its inputs**: the recompute must rebuild the
  exact forward values. RNG-using ops must derive their stream from explicit
  stored seeds — `dropout(p, seed)` qualifies by construction (its mask is a
  pure function of `(seed, element index)` and is never stored); ambient RNG
  state does not.
- **nesting is allowed**: a `checkpoint` inside a block is recomputed on a
  frame of its own inside the outer recompute (pinned bitwise against the
  plain graph by test).

Contract for `extra` (`checkpointWithContext` only): it sits between
`*ExecContext` and the inputs in the block signature, is stored **by value**
in the backward node, and is passed verbatim to both the forward run and the
recompute. It is the channel for everything non-differentiable — frozen
quantized/f16/bf16 constant tensors, RoPE tables, config values, layer
struct pointers. Anything reachable through it must remain valid until
backward completes (the node keeps only the value bits, no deep copy or
refcount) and is treated as constant: tensors reachable through `extra`
never receive gradients — trainable tensors must travel through `inputs`.
A `{}` (void) `extra` degenerates to plain `checkpoint`.

Runtime behavior and constraints:

- With no grad-requiring input, checkpoint degenerates to a no-grad forward
  (same adoption tail as any facade op). The result follows the standard
  ownership contract: caller-owned with no scope open, a `scope_owned`
  borrow otherwise.
- Each recompute runs its facade ops on a scope stack that lives in the
  recompute frame (`ExecContext.installScopeStack`), never on the context's
  own stack, so independent checkpoint nodes driven from pool threads, and
  recomputes on different contexts, proceed without any shared lock; the
  inner backward runs via `backwardGradSerial`, which confines a frame to
  the thread running it. Checkpoint nodes themselves always execute
  synchronously on the scheduling thread.
- Block runs (forward and recompute) pin the quantized-RHS dot to the CPU
  kernels for their duration (`ctx.disableQuantDotGpu()`, a per-context
  atomic depth count) so both runs take the same kernels.
- The recompute errors with `error.CheckpointOutputNotDifferentiable` if the
  re-run block's output carries no graph, and
  `error.CheckpointMissingInputGradient` if a grad-requiring input received
  none.

```zig
fn ckptLayer(
    ctx: *fucina.ExecContext,
    x: *const fucina.Tensor(.{ .batch, .in }),
    w: *const fucina.Tensor(.{ .out, .in }),
) !fucina.Tensor(.{ .batch, .out }) {
    var z = try x.dot(ctx, w, .in); // intermediates are scope-owned:
    defer z.deinit(); //             deinit is a safe no-op inside the block
    return z.tanh(ctx);
}

test "checkpointed layer backward" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, 2 }, &.{ 1, 2 });
    defer x.deinit();
    var w = try fucina.Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 1, 2 }, &.{ 0.5, -0.25 });
    defer w.deinit();

    var y = try fucina.checkpoint(&ctx, ckptLayer, .{ &x, &w });
    defer y.deinit(); // only inputs + this output are retained
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx); // block re-runs here to rebuild the subgraph

    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    // z = 0, tanh'(0) = 1 -> dL/dw = x
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, try gw.dataConst());
}
```

Frozen state through `extra`:

```zig
const Frozen = struct {
    w: *const fucina.Tensor(.{ .out, .in }), // constant: never receives grads
    alpha: f32,
};

fn frozenLayer(
    ctx: *fucina.ExecContext,
    extra: Frozen,
    x: *const fucina.Tensor(.{ .batch, .in }),
) !fucina.Tensor(.{ .batch, .out }) {
    var z = try x.dot(ctx, extra.w, .in);
    defer z.deinit();
    return z.scale(ctx, extra.alpha);
}

test "checkpointWithContext carries frozen state" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, 2 }, &.{ 1, 2 });
    defer x.deinit();
    var w = try fucina.Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 1, 2 }, &.{ 3, 4 });
    defer w.deinit(); // frozen weight rides in `extra`, not `inputs`

    var y = try fucina.checkpointWithContext(&ctx, frozenLayer, Frozen{ .w = &w, .alpha = 0.5 }, .{&x});
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1.5, 2 }, try gx.dataConst()); // alpha * w
}
```

The gradients are **bitwise identical** to the non-checkpointed forward: the
recompute runs the identical ops on the identical input views
(`src/ag/checkpoint_tests.zig` asserts parity to the bit).

## 5.6 Custom VJPs (`src/ag/custom.zig`)

```zig
pub fn customVjp(ctx: *ExecContext, comptime Spec: type, extra: anytype,
                 inputs: anytype) !Spec.Output
```

`customVjp` admits a user-defined differentiable op. (For plain elementwise
scalar functions, prefer the `elementalUnary`/`elementalBinary` convenience
tier built on this adapter — [§4.4](04-tensor-operations.md#44-unary-ops-srcagtensorzig-srcbackendopszig) — which needs no raw-tensor code.) The
public contract stays in f32 facade tensors — `inputs` is a tuple of
pointers to f32 facade tensors, `Spec.Output` an f32 facade tensor type —
while the `Spec` computes on raw tensors (`fucina.internal.RawTensor`, [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md))
inside the adapter:

```zig
const Spec = struct {
    pub const Output = fucina.Tensor(.{ ... });
    pub fn forward(ctx: *ExecContext, extra: E,
                   inputs: []const *const RawTensor) !RawTensor { ... }
    pub fn backward(ctx: *ExecContext, extra: E,
                    inputs: []const *const RawTensor,
                    output: *const RawTensor, gy: *const RawTensor,
                    needs_grad: []const bool, out: []?RawTensor) !void { ... }
};
```

Missing `Output`/`forward`/`backward` declarations, non-f32 facade types, or
a non-tuple `inputs` are compile errors. Semantics:

- `forward` returns an owned raw tensor; `customVjp` validates it against
  `Output`'s tag rank and wraps it.
- If no input requires gradients, or a `noGrad` scope is active, the result
  is a plain no-grad tensor and `backward` is never referenced at runtime.
- Otherwise the node captures refcounted **views** of every input value and
  of the output (cheap, no copies), the input `GradState` pointers as
  operands, and `extra` **by value** (same lifetime contract as checkpoint's
  `extra`: pointees must outlive backward).
- At backward time the adapter passes the saved views, the saved output, and
  the upstream `gy`; `backward` must write an *owned* raw tensor into
  `out[i]` for every true `needs_grad[i]` (the engine consumes and deinits
  them; leaving a required slot `null` surfaces as
  `error.MissingBackwardGradient`, [§5.2](05-automatic-differentiation.md#52-running-backward-srcagcorezig)). Each produced gradient is
  shape-checked against its input; a mismatch fails the pass with
  `TensorError.ShapeMismatch`.

<!-- snippet: helper -->
```zig
const RawTensor = fucina.internal.RawTensor;

const ScaledSquare = struct {
    pub const Output = fucina.Tensor(.{.d});

    pub fn forward(ctx: *fucina.ExecContext, extra: f32, inputs: []const *const RawTensor) !RawTensor {
        var sq = try ctx.elementwise(.f32, .mul, inputs[0], inputs[0]);
        defer sq.deinit();
        return ctx.scale(.f32, &sq, extra); // y = extra * x^2
    }

    pub fn backward(
        ctx: *fucina.ExecContext,
        extra: f32,
        inputs: []const *const RawTensor,
        output: *const RawTensor,
        gy: *const RawTensor,
        needs_grad: []const bool,
        out: []?RawTensor,
    ) !void {
        _ = output;
        if (needs_grad[0]) {
            var slope = try ctx.scale(.f32, inputs[0], 2 * extra); // dy/dx = 2*extra*x
            defer slope.deinit();
            out[0] = try ctx.elementwise(.f32, .mul, gy, &slope); // engine consumes out[0]
        }
    }
};
```

## 5.7 Gradient checking (`src/ag/gradcheck.zig`)

```zig
pub fn gradcheck(ctx: *ExecContext, comptime loss_fn: anytype,
                 inputs: anytype, options: Options) !Result

// root re-exports (src/ag.zig / src/fucina.zig):
pub const GradcheckOptions = gradcheck_mod.Options;
pub const GradcheckResult = gradcheck_mod.Result;
```

Central finite-difference validation of a deterministic scalar loss.
`loss_fn` must be `fn (*ExecContext, ...input ptrs) !Tensor(.{})` (one
pointer parameter per tuple entry; a non-scalar return is a compile error).
`inputs` is a tuple of **mutable** pointers to f32 facade tensors: variable
inputs are checked — the harness perturbs their owned storage element by
element, so they must be contiguous — while constants may appear and are
ignored. All inputs' gradients are zeroed before the check and again on exit
(the accumulated analytical gradients do not leak into the caller's
training state). The analytical backward and every loss evaluation run under
their own exec scope, so composed ops inside the loss are fine.

| `Options` field   | default | meaning                                        |
|-------------------|---------|------------------------------------------------|
| `eps`             | `1e-3`  | central-difference step (`f64`)                 |
| `abs_tol`         | `1e-3`  | absolute tolerance floor                        |
| `rel_tol`         | `1e-2`  | relative tolerance factor                       |
| `print_mismatch`  | `true`  | `std.debug.print` the first failing element     |

Per element the check is `|g_num − g_ana| ≤ abs_tol + rel_tol·|g_ana|`.
`Result` reports `checked` (element count), `max_abs_error`, and
`max_rel_error`. Errors: `error.InvalidGradcheckOptions` (non-finite or
non-positive `eps`; negative or non-finite tolerances), `error.NoVariableInputs`,
`error.MissingAnalyticalGradient` (backward produced nothing for a
variable), `error.GradientShapeMismatch`, and `error.GradientMismatch` (the
tolerance failure). Any error from the loss itself propagates.

```zig
fn squareLoss(ctx: *fucina.ExecContext, x: *const fucina.Tensor(.{.d})) !fucina.Tensor(.{}) {
    var y = try fucina.customVjp(ctx, ScaledSquare, @as(f32, 0.5), .{x});
    defer y.deinit();
    return y.sumAll(ctx);
}

test "customVjp validated by gradcheck" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, -2, 3 });
    defer x.deinit();

    const result = try fucina.gradcheck(&ctx, squareLoss, .{&x}, .{});
    try std.testing.expectEqual(@as(usize, 3), result.checked);
    try std.testing.expect(result.max_abs_error < 1e-2);
}
```

`gradcheck` is the oracle used throughout `src/ag/gradcheck_tests.zig` to
validate both built-in VJPs (conv2d, losses, norms) and custom ops, and by
the table-driven sibling `src/ag/gradcheck_vjp_tests.zig`, which runs one
small-shape case per remaining backward record (softmax, norms, RoPE,
matmul/bmm, gated units, reductions, functional updates, views, casts,
causal convolutions, unfold/fold); use it for every new `customVjp` spec.

## 5.8 VJP coverage inventory (`src/ag/backward/`)

Every differentiable facade op attaches a concrete VJP record from its
domain's file in `src/ag/backward/`. Coverage by family (op names as on the facade, [§4](04-tensor-operations.md)):

| Family | Differentiable ops | Notes |
|---|---|---|
| Pointwise arithmetic | `add`, `sub`, `mul`, `div`, `maximum`, `minimum`, `pow`, `scale`, `addScalar`, `subScalar`, `divScalar`, `powScalar`, `biasAdd` | broadcast operands reduce gradients back to source tags; `maximum`/`minimum` split the gradient evenly on exact ties; `biasAdd`'s slice bias is constant |
| Selection / masking | `where` (grads to both value operands), `maskedFill` (grad zeroed where filled), `clamp`, `dropout`, `bandPart`, `tril`, `triu` | `cond`/`mask` are non-grad; dropout regenerates its mask from `(seed, index)` in forward, backward, and recompute; the triangular family multiplies by a constant 0/1 band plane, so its VJP is the exact same-band mask |
| Unary activations | `relu`, `leakyRelu`, `exp`, `sqrt`, `rsqrt`, `sigmoid`, `silu`, `log`, `log1p`, `neg`, `abs`, `sin`, `cos`, `tanh`, `fastTanh`, `softcap(cap)`, `gelu`, `quickGelu`, `elu`, `geluErf`, `floor`, `ceil`, `round`, `sign`, `reciprocal`, `unary`, `snake`, `prelu` | `prelu`/`snake` also differentiate their parameters; `floor`/`ceil`/`round`/`sign` have zero gradient a.e. (torch convention) |
| Gated units | `gated`, `glu`, `swiglu`, `geglu`, `situ`, `splitGated` | fused split+gate VJPs; `situ` differentiates both the soft-clamped up input and its soft-bounded SiLU gate |
| Reductions / statistics | `sum`, `sumMany`, `sumAll`, `mean`, `variance`, `prod`, `cumsum`, `cumprod`, `linearRecurrence`, `logsumexp`, `standardizeAxis`, `norm`, `normAll`, `max`, `min`, `topK` (values arm), `sort` (values arm) | `max`/`min` route gradient to the first extremum (strict tie-break); `topK`/`sort` values scatter back through the saved indices; `linearRecurrence` differentiates input, decay, and initial state through one reverse scan ([§4.7](04-tensor-operations.md#47-reductions-and-scans-srcagtensorzig)) |
| Structure / views | `withTags`, `permuteTo`, `transpose`, `alignTo`, `insertAxis`, `squeeze`, `split`, `merge`, `reshape`, `viewWithStrides`, `materialize`, `contiguous`, `broadcastTo`, `flatten`, `narrow`, `select`, `slice`, `sliceStep`, `pad`, `zeroPad2d`, `constantPad2d`, `concat`, `stack`, `unbindInto`, `repeatAxis`, `flip`, `roll`, `rollBy`, `shiftBy`, `diagonal`, `trace`, `diag`, `diagEmbed`, `gather`, `indexSelect`, `takeAlongAxis`, `indexAdd`, `scatterAdd`, `scatter`, `maskedSelect`, `maskedScatter`, `setSlice`, `setRows`, `zeroSlice`, `zeroRows`, `relposShift`, `to` (f32/f16/bf16 targets, [§3.8](03-tensors-types-construction-and-data-access.md#38-casting-todtype-srcagtensorzig-srcexecconvertzig)) | view VJPs scatter through the saved layout; `detach` deliberately cuts the graph |
| Norms / softmax | `softmax` (all fused options; `.mask` must not require grad), `logSoftmax`, `rmsNorm`, `rmsNormMul`, `rmsNormMulAdd`, `rmsNormMulRopeHalfPrepared`, `layerNorm` (plain + affine), `groupNorm`, `l2Normalize`, `cosineSimilarity` | |
| Losses | `crossEntropy`, `crossEntropy`, `linearCrossEntropy`, `linearDistill`, `mseLoss`, `huberLoss`, `bceLoss`, `klDivLoss`, `nllLoss` | `linearCrossEntropy` differentiates both the input and the classifier weight without materializing the logit gradient ([§4.15](04-tensor-operations.md#415-losses-and-similarity-srcagtensorzig-srcexeclosszig)) |
| Contractions | `dot` (f32×f32: both operands; quantized RHS: lhs-only, the RHS is a frozen constant; f16/bf16 RHS: lhs always, plus an f32 dW when the RHS is a grad-requiring 16-bit variable), `einsum` (f32×f32: both operands; f16/bf16 RHS: same variable-RHS contract as dot; each gradient is itself an einsum — GEMM-lowered for every tag structure, broadcast over forward-summed axes; `DotBackward`/`ConstRhsDotBackward` delegate to the einsum records), `addDot` (the fused addmm: all three operands — the base gradient is the upstream gradient itself, shared as a view; the `a`/`b` gradients are the dot VJP contractions, with the same internal branch split as `dot`), `einsumMany` (composes binary einsum records), `matmul` (2-D GEMM `.plain`/`.trans_b`, batched bmm all kinds; rank-2 `.trans_a` is a compile error directing to `dot`), `dotTernarySte` (straight-through estimator: dx through the quantized weight, dW as-if-unquantized) | |
| Convolutions / pooling | `conv1d`, `convTranspose1d`, `causalConv1d`, `groupedCausalConv1d`, `causalDepthwiseConv1d`, `conv2d`, `conv2dRelu`, `maxPool2d`, `avgPool2d`, `upsample2xNearest`, `unfold`, `fold`, `channelAffine` | `conv2d` differentiates input, weight, and bias; `conv2dRelu` falls back to the composed `conv2d` + `relu` path when any operand requires grad; `unfold`/`fold` are exact adjoints of each other (im2col/col2im) |
| Position / attention | `rope` (table and on-the-fly sources, both modes), `groupedAttention` | attention grad matrix: f32 KV = full q/k/v; f16 or q8_0 KV = q-only (caches are constants); `.bias` or multi-stream KV = inference-only (`error.UnsupportedGradient`) |

Intentionally no-grad (result is a constant; grad-requiring operands are
either irrelevant or rejected):

- **Constant results by nature**: `argmax`, `argsort`, `multinomial`, the
  `indices` arm of
  `topK` and `sort`, `compare`, `isnan`, `isinf`, `isfinite`, `any`, `all`,
  `anyAll`, `allAll`, `logicalAnd`, `logicalOr`, `logicalXor`,
  `logicalNot`, `bandMask` — index/mask/sampling outputs; gradients are
  undefined.
- **Inference-only packed kernels** — dense `packRhs`/`dotPacked` fail with
  `error.GradientPackedMatmulUnsupported`; quantized `dotPacked` and the fused
  `rmsNormMulDotPacked`/`splitSwiGluDotPacked`/`gegluQuantDotPacked` fail with
  `error.GradientQuantizedMatmulUnsupported` when an operand requires grad
  ([§10](10-quantization.md)). For a *trainable* path use ordinary dense `dot`, or quantized `dot`
  (lhs-grad) / `dotTernarySte` as appropriate.
- **Prepared-conv entries** — fail with
  `error.GradientPreparedConv2dUnsupported` when an operand requires grad:
  `prepareConv2dWeights`, `conv2dPrepared`, `conv2dPreparedRelu` ([§4.14](04-tensor-operations.md#414-convolution-and-channel-last-vision-ops-srcagtensorzig) —
  the prepared Winograd planes live outside the graph; use `conv2d`/
  `conv2dRelu` for the trainable path).
- **In-place / storage-consuming helpers** — fail with
  `error.UnsupportedGradient`: `addAxisVectorInPlace`,
  `addAxisVectorUnaryInPlace`, `addScaledInPlace`, `takeAddNoGrad`,
  `takeScaleNoGrad`, `routerTopK`.
- **Casts off the float seam**: `to` with a target other than `.f32`,
  `.f16`, or `.bf16` rejects grad-carrying inputs with
  `error.GradientCastUnsupported` (`to(.f32)` and the f16/bf16 narrows are
  differentiable, [§3.8](03-tensors-types-construction-and-data-access.md#38-casting-todtype-srcagtensorzig-srcexecconvertzig)).
- The typed and quantized constant tensor branches ([§3](03-tensors-types-construction-and-data-access.md), [§10](10-quantization.md)) never carry
  gradients at all.
