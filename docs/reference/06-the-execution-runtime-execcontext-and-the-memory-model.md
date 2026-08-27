# 6. The execution runtime: ExecContext and the memory model

`fucina.ExecContext` is the eager runtime boundary. Every op call, every
tensor allocation, and every gradient pass goes through one context: it owns
the allocator wrapper, the backend instance, the transient-buffer pool, the
lazily-created worker team, and the exec-scope stack. There is no graph
object and no deferred execution — an op call runs the kernel and returns an
owned result immediately. This section covers the context itself and the
memory model it implements; the op surface it exposes is catalogued in [§4](04-tensor-operations.md),
autograd semantics in [§5](05-automatic-differentiation.md), backend selection in [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md).

## 6.1 ExecContext: role and lifecycle (`src/exec.zig`, `src/exec/runtime.zig`)

`ExecContext` is one struct: the runtime substrate is the embedded
`rt: Runtime` (allocation, thread, and scope machinery; the `Runtime`
struct is declared in `src/exec/runtime.zig`), the trailing fields are
model/session execution state (kernel pinning, MoE-decode scratch), and
every method is an alias line resolving to a function that takes
`*ExecContext` first. The substrate functions (lifecycle, scopes, worker
team, tensor allocation primitives, `replace`) live in
`src/exec/runtime.zig`; the domain ops live in the modules under `src/exec/`
(`elementwise.zig`, `matmul.zig`, `conv.zig`, `attention.zig`, `moe.zig`, …)
and receive `*ExecContext` explicitly.

```zig
pub const Runtime = struct {                 // src/exec/runtime.zig
    thread_safe_allocator: thread.ThreadSafeAllocator,
    allocator: Allocator,            // fat pointer into thread_safe_allocator
    parallel_pool: std.atomic.Value(?*thread.Pool), // the published worker team (pc() snapshots it)
    buffers: BufferPool,
    tuning: tuning.Overrides,
    work_pool: thread.Pool,          // + work_pool_ready, work_pool_failed, work_pool_mutex
    dot_backward_worker: thread.OneShotWorker, // + ready flag, mutex
    scopes: ScopeStack,              // the context's own exec-scope stack
    fp_env_at_init: ?fpenv.Environment,
};

pub const ExecContext = struct {
    rt: Runtime,                     // the runtime substrate
    quant_dot_gpu_disabled: std.atomic.Value(u32), // open disableQuantDotGpu scopes
    rowwise_numerics_pinned: u32, // open pinRowwiseNumerics scopes
    moe_scratch: MoeDecodeScratch,   // grow-only MoE decode scratch

    pub fn allocator(self: *const ExecContext) Allocator // rt.allocator, the public accessor
    pub const init = exec_runtime.init;      // (self: *ExecContext, allocator: Allocator) void
    pub const deinit = exec_runtime.deinit;  // (self: *ExecContext) void
    pub const add = exec_elementwise.add;    // ... one alias line per op
};
```

Public code takes the allocator through `ctx.allocator()`; the other
substrate fields are internal but observable, reached as `ctx.rt.<field>`:

| Field | Type | Created |
|---|---|---|
| `rt.thread_safe_allocator` | `thread.ThreadSafeAllocator` | at `init` (wraps the caller's allocator in a mutex) |
| `rt.allocator` | `std.mem.Allocator` | at `init` (fat pointer into `rt.thread_safe_allocator`) |
| `rt.parallel_pool` | `std.atomic.Value(?*thread.Pool)` | at `init` (null); published by `tryWorkPool` ([§6.6](06-the-execution-runtime-execcontext-and-the-memory-model.md#66-the-worker-team-srcthreadzig-srcparallelzig), [§9.2](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#92-the-kernel-interface-and-the-kernel-contract-srcbackendzig-srcbackendnativezig)) |
| `rt.buffers` | `BufferPool` | at `init` ([§6.5](06-the-execution-runtime-execcontext-and-the-memory-model.md#65-bufferpool-transient-reuse-and-scratch-leases-srcexecbuffer_poolzig)) |
| `rt.work_pool` | `thread.Pool` | lazily, on first `tryWorkPool` ([§6.6](06-the-execution-runtime-execcontext-and-the-memory-model.md#66-the-worker-team-srcthreadzig-srcparallelzig)) |
| `rt.dot_backward_worker` | `thread.OneShotWorker` | lazily, on first `dotBackwardWorker` ([§6.6](06-the-execution-runtime-execcontext-and-the-memory-model.md#66-the-worker-team-srcthreadzig-srcparallelzig)) |
| `rt.scopes` | `ScopeStack` (entries + open depth) | at `init`, empty ([§6.3](06-the-execution-runtime-execcontext-and-the-memory-model.md#63-exec-scopes-implicit-ownership-for-training-srcexeczig-srcexecruntimezig)) |
| `quant_dot_gpu_disabled` | `std.atomic.Value(u32)` | at `init`, zero; the open `disableQuantDotGpu` scopes ([§5.5](05-automatic-differentiation.md)) |

**The init(self-pointer) pattern.** `init` takes `self: *ExecContext` and
returns `void` instead of returning a value: the context is self-referential
(`ctx.rt.allocator` is a fat pointer into `ctx.rt.thread_safe_allocator`), so it
must be initialized in place at its final address and must never be moved or
copied afterwards. The PINNED property is stated on `Runtime`, the field
that embeds it. The idiom:

```zig
test "context lifecycle" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc); // in place: the context is self-referential, never move it
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var y = try x.scale(&ctx, 2.0);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8 }, try y.dataConst());
}
```

`init` cannot fail (it allocates nothing). `deinit` tears down in order:
MoE scratch, then the one-shot worker and the worker team (if they were ever
created; `parallel_pool` is reset to null first), then any exec scopes still open
(defensive release), then the buffer pool. `BufferPool.deinit` asserts that
no pooled buffer is still outstanding — a tensor leaked past context teardown
fails this assertion in safety builds rather than silently leaking. After
`deinit` the struct is `undefined`.

**Substrate methods** (everything else on `ExecContext` is an op, see [§4](04-tensor-operations.md);
the signatures below are the `src/exec/runtime.zig` functions the struct
aliases):

```zig
pub fn execScopeActive(self: *const ExecContext) bool
pub fn openExecScope(self: *ExecContext) ExecScope
pub fn closeExecScope(self: *ExecContext, mark: ExecScope) void
pub fn adopt(self: *ExecContext, entries: []const ScopeEntry) !void
pub fn bufferEntry(comptime dtype: DType, buffer: *BufferOf(dtype)) ScopeEntry
pub fn tryWorkPool(self: *ExecContext) !*thread.Pool
pub fn workPool(self: *ExecContext) ?*thread.Pool
pub fn pc(self: *const ExecContext) backend.ParallelConfig
pub fn dotBackwardWorker(self: *ExecContext) ?*thread.OneShotWorker
pub fn pinRowwiseNumerics(self: *ExecContext) RowwiseNumericsScope
pub fn rowwiseNumericsPinned(self: *const ExecContext) bool
pub fn setTuning(self: *ExecContext, overrides: tuning.Overrides) void
pub fn replace(self: *ExecContext, old: anytype, new_value: anytype) @TypeOf(new_value)
pub fn broadcastTo(self: *ExecContext, x: *const Tensor, shape: anytype) !Tensor
```

There is no layout-classification entry point: each lowering inspects
its operands' strides itself (`isContiguous`, and for elementwise ops
`tailBroadcastInfo` in `src/exec/elementwise.zig`, which recognises a
scalar or bias-style operand broadcast over the leading axes) and falls back
to `prepareContiguous` for anything else. `broadcastTo`
returns a zero-copy view (a refcounted alias, [§6.2](06-the-execution-runtime-execcontext-and-the-memory-model.md#62-the-memory-model-who-owns-an-op-result-docsmemory-modelmd)); `shape` is either a
`[rank]usize` array/tuple (comptime rank) or a `[]const usize` slice.
`setTuning` installs per-context tuning overrides (`fucina.tuning.Overrides`,
[§2.6](02-toolchain-build-and-project-wiring.md#26-runtime-environment-variables)): fields left null follow the process-wide FUCINA_* gates, so two
contexts in one process can run different route policy (first consumer: the
CPU f32 weight-shadow route).
`pinRowwiseNumerics()` opens a scope that pins every batched quant-matmul
entry on the context to the `m == 1` kernel numerics
(`QuantMatmul.numerics = .rowwise`, forced context-wide) — the
packed/plain entries run as independent single-row calls of themselves,
the fused K-quant entries keep their per-row tail kernels for every row,
and the batched MoE op skips the lane-packed Q8_Kx4 kernels — so a
speculation verify batch produces logits bit-identical to sequential
decode, the property that keeps deep speculative drafting lossless
([§13.9](13-the-model-stack-fucina_models.md#139-speculative-decoding-srcmodelstextspeculative)). Batch matmul throughput is
deliberately sacrificed while pinned; open the scope around the verify
forward only and `close()` it after (scopes nest — the backing field is
`rowwise_numerics_pinned`, an open-scope count, zero at `init`;
`rowwiseNumericsPinned()` is the query). The
scope and pool methods are covered below.

**MoE decode scratch** (`moe_scratch`, ops in `src/exec/moe.zig`). A
grow-only, mutex-guarded scratch region backing the single-row MoE decode
ops: the per-token region sizes are model constants, so after the first
token the hot path performs no allocations — one uncontended lock instead of
several allocator/pool round-trips per layer. The discipline is
`lockMoeDecodeScratch()` … carve … run … `unlockMoeDecodeScratch()`, holding
the lock for the whole op because the expert tasks write into the carved
slices. `carveMoeDecodeScratch(QgBlock, Task, hidden_blocks, top_k, out_pe,
hidden, blocks_per_g)` returns a `MoeDecodeScratchView(QgBlock, Task)` —
borrowed slices carved from the region (`qx` Q8_K activation blocks,
`gate_buf`/`up_buf`/`g_buf`, `qg`, `outs`, `tasks`), valid only while the
lock is held; `carveMoeDecodeChainScratch` adds a `states` slice and a
caller-sized task count for dependency-chained decode
(`MoeDecodeChainScratchView`). Every carved type must align to ≤ 8 (compile
error otherwise). This is the seam in-tree LLM-band code uses to build
custom MoE decode paths (`src/models/gemma/moe.zig`, [§13](13-the-model-stack-fucina_models.md)); `deinit` frees the
scratch with the context.

## 6.2 The memory model: who owns an op result (`docs/MEMORY-MODEL.md`)

The contract, in one sentence: **every tensor an op returns is owned by the
caller and must be deinitialized exactly once — unless an exec scope is open,
in which case op results are scope-owned borrows and `deinit` on them is a
safe no-op.** The full rationale (including why a generic arena allocator was
evaluated and rejected) is recorded in [MEMORY-MODEL.md](../MEMORY-MODEL.md);
this subsection restates the operative rules.

Ownership by construction source:

| Tensor came from | Owner | `deinit` required? |
|---|---|---|
| any facade op result, **no scope open** | caller | yes, exactly once |
| any facade op result, **scope open** | innermost exec scope | no (safe no-op); never use after the scope closes |
| explicit constructors: `variable`, `constant`, `fromSlice`, `fromTensor`, `empty`, `zeros`, `ones`, `full`, `scalar`, … ([§3](03-tensors-types-construction-and-data-access.md)) | caller | yes — even inside a scope |
| fetched gradients: `grad`, `gradView` ([§5](05-automatic-differentiation.md)) | caller | yes — even inside a scope |
| `ctx.*` raw construction helpers ([§6.4](06-the-execution-runtime-execcontext-and-the-memory-model.md#64-raw-construction-and-copy-helpers-on-ctx-srcexeczig-srcexecruntimezig)) | caller | yes — never scope-adopted |
| typed/quantized-constant tensor results ([§3](03-tensors-types-construction-and-data-access.md), [§10](10-quantization.md)) | caller | yes — typed ops are not scope-adopted |

**`deinit` is the recycling driver, not a naive free.** The chain is
`tensor.deinit()` → `buffer.release()` (atomic refcount decrement) → refcount
hits 0 → the buffer's release hook returns it to the `BufferPool` free list.
A released transient is immediately reusable by the next op — same-sized
successive allocations get the *same address* back, which keeps the hot
working buffer warm in cache. This is asserted behavior:

```zig
test "deinit recycles transient buffers through the pool" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try ctx.fromSlice(.f32, &.{3}, &.{ 1, 2, 3 });
    defer a.deinit();

    var first = try ctx.elementwise(.f32, .add, &a, &a);
    const first_ptr = first.dataConst().ptr;
    first.deinit(); // storage returns to the pool free list

    var second = try ctx.elementwise(.f32, .add, &a, &a); // same size: the pool hands back the same address
    defer second.deinit();
    try std.testing.expectEqual(first_ptr, second.dataConst().ptr);
}
```

**Two lifetime regimes.** Inference tensors are constants
(`grad_state == null`); their `deinit` releases storage immediately, so the
pool behaves arena-like *within* a forward pass — the idiomatic
`var x = try someOp(ctx, ...); defer x.deinit();` gives an O(1) working set.
Training variables retain their inputs: backward functions store operand
*views* at op-execution time, and a view bumps the storage refcount, so those
buffers cannot return to the pool until the tape node is destroyed in/after
`backward()`. Holding activations for backward is inherent to training, not
pool overhead.

**Views are refcounted aliases.** Every view operation (`cloneView`,
`reshape`, `broadcastTo`, `narrow`, strided views — [§3](03-tensors-types-construction-and-data-access.md), [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)) retains the
source buffer and releases it on its own `deinit`; a view's lifetime is
independent of its parent's. Deinitializing a source tensor while views on it
live is safe — the storage survives until the last reference drops. This is
also why the runtime cannot use region-reset arenas: a zero-copy `narrow`
into a session-lifetime KV cache ([§13](13-the-model-stack-fucina_models.md)) has a per-object lifetime no region
reset can express.

**The carried-value pattern: `ctx.replace`.** Residual streams and other
accumulators advance a single binding through many ops. `replace`
deinitializes the old value and returns the new one in one statement:

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

`new_value` must be an error union of `@TypeOf(old)` (compile error
otherwise). On error the old value is *not* consumed — the caller's binding
and `defer`/`errdefer` arms stay valid and the error propagates. On success
the old value is released (one reference) and the new value returned for
rebinding. Inside an exec scope the release is a safe no-op on scope-owned
results, so the same forward code is training-safe. `replace` is generic
over any owned value with a `deinit` method (tagged tensors, projection
structs).

**Why not an arena.** Summarizing MEMORY-MODEL.md [§4](04-tensor-operations.md): a per-forward reset
arena would (i) balloon peak memory from the working set (~6–12 live
transients per block) to the sum of all intermediates, (ii) destroy the
address-reuse cache locality shown above, (iii) be unable to express
refcounted views and KV-cache aliasing, and (iv) be incorrect for training,
where activations must outlive the forward. The `BufferPool` already
delivers allocation amortization — the only real arena advantage — plus
intra-pass reuse and a bounded cap.

## 6.3 Exec scopes: implicit ownership for training (`src/exec.zig`, `src/exec/runtime.zig`)

Training does not break deinit-ASAP: each differentiable result's
`GradState` is reference-counted and every consumer record retains its
operands ([§5](05-automatic-differentiation.md), [TRAINING.md §2](../TRAINING.md)), so releasing an
intermediate handle before `backward()` is always safe. Exec scopes make the
context the owner of op results on top of that, so a training forward needs
no handle bookkeeping at all.

```zig
pub const ExecScope = struct { index: usize };
pub const ScopeRelease = *const fn (*anyopaque) void;
pub const ScopeEntry = struct { ptr: *anyopaque, release: ScopeRelease }; // one reference: a value buffer of any dtype, or a graph node

pub fn openExecScope(self: *ExecContext) ExecScope
pub fn closeExecScope(self: *ExecContext, mark: ExecScope) void
pub fn execScopeActive(self: *const ExecContext) bool
```

Semantics:

- **While a scope is open, every tensor returned by a facade op, of any
  dtype, is adopted by the innermost scope.** The value the caller receives
  is a borrow with its `scope_owned` flag set: `deinit` on it is a safe
  no-op for the value and the graph node alike (arena-style), and using it
  after the scope closes is use-after-free. Adoption is wired into the op
  tails (`finishOp` / `finishNoGrad` / `finishTyped` in
  `src/ag/tensor/plumbing.zig`) and covers differentiable results, no-grad
  f32 results (eval on constants, the `values` arm of `topK`/`sort`, …),
  the i64 index outputs (`argmax`, `argsort`, the `indices` arms), `.bool`
  masks, 16-bit casts, and the quantized branch's views and row gathers:
  one rule, an op result under a scope is a borrow.
- **What stays caller-owned even inside a scope:** tensors created
  explicitly (`variable`, `constant`, `fromSlice`, and the other [§3](03-tensors-types-construction-and-data-access.md)
  constructors), fetched gradients (`grad` / `gradView`), and the raw
  `ctx.*` construction helpers of [§6.4](06-the-execution-runtime-execcontext-and-the-memory-model.md#64-raw-construction-and-copy-helpers-on-ctx-srcexeczig-srcexecruntimezig).
- **`closeExecScope(mark)` releases every reference adopted since `mark`,
  newest first**: one entry per value buffer and one per graph node, each
  through its own release call. Every borrow is invalid from then on (its
  `deinit` stays a no-op). Close a scope after any `backward()` over its
  results.
- **Scopes manage lifetimes only.** The graph is reference-counted on its
  own ([§5](05-automatic-differentiation.md): a record retains its operands' states), so a scope is never
  required for correctness: unscoped code that releases every handle it
  holds trains identically. The scope removes the handle bookkeeping.
- **Scopes nest with strict stack discipline** — close in reverse order of
  opening. A nested scope releases only its own suffix; values adopted by an
  outer scope survive an inner close.
- **Error safety for free:** if an op fails mid-forward, the scope already
  owns the prefix of results, so model code inside a scope needs no
  `errdefer` chains.
- **A stack is not thread-safe:** open/close and the ops between them run
  on one thread, like every other context mutation ([§6.9](06-the-execution-runtime-execcontext-and-the-memory-model.md#69-the-thread-safety-contract)).
  The scope functions route to the context's own `scopes` stack unless the
  calling thread has installed a stack of its own (`ScopeStack`,
  `installScopeStack(&stack)` / `restoreScopeStack(previous)`): the
  checkpoint recompute in backward does exactly that, so a recompute driven
  from a pool thread adopts into its own frame and never touches the
  context's stack ([§5.5](05-automatic-differentiation.md)).

The canonical training-step pattern — a per-iteration scope, no keeps, no
defers on intermediates:

```zig
test "training step under an exec scope" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var w = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer w.deinit(); // parameters stay caller-owned

    for (0..2) |_| {
        const scope = ctx.openExecScope();
        defer ctx.closeExecScope(scope); // releases all adopted intermediates, newest first
        const y = try w.mul(&ctx, &w); // scope-owned borrow: no defer needed
        var loss = try y.sumAll(&ctx);
        try loss.backward(&ctx);

        var gw = (try w.grad(&ctx)).?; // fetched gradients stay caller-owned
        defer gw.deinit();
        try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6 }, try gw.dataConst());
        w.zeroGrad();
    }
}
```

**Write-once forward code.** Because `deinit` on a scope-owned result is a
no-op, defer-deinit forward code — the inference idiom, including
`ctx.replace` for residual streams — runs unchanged under a scope. Write the
forward once; train it by opening a scope around it:

```zig
test "deinit on a scope-owned result is a safe no-op" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer x.deinit();

    const scope = ctx.openExecScope();
    defer ctx.closeExecScope(scope);
    var y = try x.add(&ctx, &x);
    defer y.deinit(); // no-op: the scope owns y — the same code runs unscoped
    try std.testing.expectEqualSlices(f32, &.{ 6, 8 }, try y.dataConst());
}
```

Scope-related errors (recoverable, not panics):

- `error.ActiveExecScopeUnsupported` — the storage-consuming
  `takeAddNoGrad` / `takeScaleNoGrad` ([§4](04-tensor-operations.md)) refuse a scope-owned operand:
  consuming a borrow would double-free at close.

**Scopes are a training tool, not an inference optimization.** A held scope
inverts the pool discipline: with deinit-ASAP a chain of same-shaped ops
recycles ~2 pooled buffers (O(1) working set, warm addresses), while a scope
keeps every intermediate live until close (O(N), cold addresses) — measured
2 vs 32 distinct buffers on a 32-op chain
([MEMORY-MODEL.md](../MEMORY-MODEL.md) [§5](05-automatic-differentiation.md); the behavior is test-pinned in
`src/ag/tensor_tests/control.zig`). For pure inference, deinit-ASAP with no scope is
the discipline; scopes are correct where holding the graph *is* the
semantics (training), and harmless on cold no-grad paths.

**Extension point.** Op implementers (e.g. `fucina.customVjp`, [§5](05-automatic-differentiation.md)) adopt
through one fallible call: `adopt(entries)` takes the handle's references,
all or nothing, as `ScopeEntry{ ptr, release }` values (`bufferEntry(dtype,
buffer)` for a value buffer of any dtype; the autograd facade packages a
graph node the same way, `core.scopeEntry`), performed before the value is
handed to the returned handle, so nothing has to be un-consumed on the
error path. The exec layer itself knows nothing about autograd types.

## 6.4 Raw construction and copy helpers on ctx (`src/exec.zig`, `src/exec/runtime.zig`)

These methods build *raw* tensors — the internal, tag-free, no-grad tensor
type ([§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md); deliberately not exported at the `fucina` root). Application code
normally uses the tagged facade constructors of [§3](03-tensors-types-construction-and-data-access.md), which wrap these; the
raw helpers appear in public signatures wherever a facade constructor takes
a `RawTensor` (e.g. `Tensor(spec).variable(&ctx, try ctx.fromSlice(.f32, ...))`)
and throughout runtime-extension code. Results are **always caller-owned and
never scope-adopted**; pair each with `deinit` (or hand ownership to a
facade constructor, which consumes the value on success and leaves it with
the caller on error).

Every constructor takes a leading comptime `DType` and a `shape: anytype`:
a `[rank]usize` array or a tuple of sizes carries the rank at comptime, a
`[]const usize` slice takes the runtime-rank arm. Allocation
(uninitialized / filled), all pool-backed:

| Function | Result | Notes |
|---|---|---|
| `empty(dtype, shape)` | `TensorOf(dtype)`, uninitialized | non-f32 dtypes route to the slab arm ([§6.5](06-the-execution-runtime-execcontext-and-the-memory-model.md#65-bufferpool-transient-reuse-and-scratch-leases-srcexecbuffer_poolzig)) |
| `zeros(dtype, shape)` | zero-filled | |
| `ones(dtype, shape)` | one-filled | |
| `full(dtype, shape, value)` | filled with `value: Scalar(dtype)` | |
| `scalar(dtype, value)` | shape `{1}` | |

Copy-in from caller data:

| Function | Semantics |
|---|---|
| `fromSlice(dtype, shape, values)` | copy `[]const Scalar(dtype)` into pooled storage |
| `fromStorageSlice(dtype, shape, values)` | copy `[]const Storage(dtype)` (block-quantized payloads, [§10](10-quantization.md)) |
| `fromBorrowedSlice(dtype, shape, values)` | **zero-copy** wrap of caller-owned `[]Scalar(dtype)`; the tensor borrows — keep the slice alive and unmoved until the tensor's `deinit`, which frees only the header |
| `fromBorrowedStorageSlice(dtype, shape, values)` | zero-copy wrap of `[]Storage(dtype)`, same borrow contract |

Copy/materialize existing tensors:

| Function | Semantics |
|---|---|
| `materialize(dtype, x)` | contiguous pooled copy of a (possibly strided/broadcast) view |
| `clone(dtype, x)` | alias for `materialize` |

Errors: `TensorError.InvalidDataLength` when `values.len` does not match the
shape's element count; `TensorError.InvalidShape` for rank 0, rank above the
max, or any zero dimension (and, for block-quantized dtypes, an innermost
dimension not divisible by the block size); `error.Overflow` on element-count
overflow; `error.OutOfMemory` from the pool. Borrowed wraps allocate only a
buffer header and never enter the pool's free lists.

```zig
test "fromSlice copies; fromBorrowedSlice wraps caller storage" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var source = [_]f32{ 1, 2, 3, 4 };
    var copied = try ctx.fromSlice(.f32, .{ 2, 2 }, &source); // pooled copy, caller-owned
    defer copied.deinit();
    source[0] = 99;
    try std.testing.expectEqual(@as(f32, 1), copied.dataConst()[0]);

    var borrowed = try ctx.fromBorrowedSlice(.f32, .{ 2, 2 }, source[0..]); // zero-copy
    defer borrowed.deinit(); // frees only the header; `source` stays caller-owned
    try std.testing.expectEqual(@as(f32, 99), borrowed.dataConst()[0]);
}
```

Internal substrate helpers (aliased on `ExecContext` for the domain
modules; not part of the op surface): `dispatchRange` /
`dispatchRangeCapped`, the `enableNative*PoolForWork` pool gates, and
`prepareContiguous(dtype, x)` returning `PreparedTensorOf(dtype)`
(`PreparedTensor` is the f32 instantiation) — a borrowed-or-owned union
whose `deinit` is a no-op on the borrowed arm, so hot paths can
`defer prepared.deinit()` unconditionally.

The dtype policy has its two helpers next to it: `prepareAs(dtype, compute,
x)` returns `PreparedTensorOf(compute)`, the contiguous borrow (or copy) when
the storage dtype is the compute dtype and one widening `cast` otherwise;
`storeAs(compute, out, value)` returns `TensorOf(out)`, the value itself when
the dtypes agree and one narrowing cast otherwise (the compute-dtype value is
released). An exec op whose kernel is f32-only writes its body once against
these two calls and serves f16/bf16 inputs by `dtype_mod.computeDType(.widened,
dtype)` ([§8.3](08-data-types-storage-and-the-raw-tensor-layer-internal.md#83-float-computeoutput-dtype-policy-srcdtypezig)).

**The raw op surface and its naming grammar.** Beyond these constructors,
`ExecContext` carries the full raw op surface — one spelling per op in
`src/exec.zig`. Ops whose storage is dtype-generic take an explicit
comptime `DType` first (`add(.f32, rank, a, b)`, `cast(.f32, .f16, x)`,
`sumAxis(.f32, rank, x, axis)`); ops whose kernels monomorphize on rank or
axis keep those comptime parameters (`softmax(rank, x, axis)`,
`conv1d(rank, ...)`). The surviving suffixes each mark a real contract
difference: `*Backward*` entries are the VJP kernels `src/ag/backward/`
dispatches to (`conv2dBackwardInput`, `dropoutBackward`,
`splitSwiGluBackward`); `*InPlace` mutates its target; `*Masked` computes
different (mask-selected) math; `*Prepared` consumes precomputed weights;
`*WithTable` / `*Into` encode an extra argument (a precomputed table, a
caller-supplied output). A whole-tensor and an axis form of the same
reduction are distinct ops (`sum` vs `sumAxis`, `mean` has only `meanAxis`).
This is the surface `customVjp` forward/backward specs ([§5.6](05-automatic-differentiation.md#56-custom-vjps-srcagcustomzig)) are written
against. `src/exec.zig` is the source of truth for the names; the domain
modules under `src/exec/` the alias lines resolve to are not public API.

## 6.5 BufferPool: transient reuse and scratch leases (`src/exec/buffer_pool.zig`)

`BufferPool` (re-exported as `exec.BufferPool`; one instance per context at
`ctx.rt.buffers`) recycles owned, refcounted storage buffers across ops.
Kernels never allocate — the allocation primitives of [§6.4](06-the-execution-runtime-execcontext-and-the-memory-model.md#64-raw-construction-and-copy-helpers-on-ctx-srcexeczig-srcexecruntimezig) are the
only source of transient tensors, and all of them draw from the pool. Two
arms share one byte budget:

- **The f32 arm** — a free list of `*storage.Buffer`, serving every
  default-dtype tensor (`acquire(len)`). In an LLM forward essentially all
  transient activations are f32, so this arm covers the hot path.
- **The byte-slab arm** — a free list of 64-byte-aligned, 4096-byte-rounded
  raw slabs serving every other storage dtype (`acquireTyped(dtype, len)`
  wraps a slab in a typed buffer header whose release hook returns the slab)
  plus non-DType packed block scratch via `acquireScratch(T, len)`. Slabs are
  reused across dtypes: an f16 slab released by one op can serve q8_k
  scratch in the next.

```zig
pub const slab_align = 64;          // covers max element alignment + cache line
pub const slab_size_quantum = 4096; // slab byte-size rounding

pub const BufferPool = struct {
    pub fn init(allocator: Allocator) BufferPool
    pub fn deinit(self: *BufferPool) void  // asserts outstanding == 0
    pub fn acquire(self: *BufferPool, len: usize) !*storage.Buffer
    pub fn acquireTyped(self: *BufferPool, comptime dtype: storage.DType, storage_len: usize) !*storage.BufferOf(dtype)
    pub fn acquireScratch(self: *BufferPool, comptime T: type, len: usize) !ScratchLease(T)
    pub fn cachedBuffers(self: *BufferPool) usize
    pub fn cachedSlabs(self: *BufferPool) usize
    pub fn cachedBytes(self: *BufferPool) usize
    pub fn outstandingBuffers(self: *const BufferPool) usize
};
```

Behavior users should know:

- **Size rounding.** f32 requests round to the next power of two up to 1024
  elements, then to the next 1024-element multiple; slab requests round to
  the next 4096-byte multiple. Rounding collapses nearby sizes into shared
  buckets, which helps reuse; a handed-back buffer may be larger than asked
  (tensors use the shape-covered prefix).
- **First-fit over an ascending free list.** `acquire` returns the smallest
  cached buffer whose capacity fits; on a miss it allocates fresh (releasing
  the pool mutex first). Releases insert *before* existing same-length
  entries, so within a size class reuse is LIFO — the most recently released
  buffer is handed back first. Same-size acquire/release cycles return the
  same address — the cache-locality property of [§6.2](06-the-execution-runtime-execcontext-and-the-memory-model.md#62-the-memory-model-who-owns-an-op-result-docsmemory-modelmd).
- **Bounded retention.** `max_cached_bytes` (default 1 GiB, shared by both
  arms) caps *cached* (free-list) bytes, not live bytes: a released buffer
  that alone exceeds the cap, or would push the cache over it, is destroyed
  instead of cached. Steady-state retention is bounded by the actual peak
  transient set of the workload.
- **Leak detection.** An atomic `outstanding` counter tracks live pooled
  buffers; `BufferPool.deinit` (run by `ExecContext.deinit`) asserts it is
  zero.
- **Scratch leases.** `acquireScratch(T, len)` returns a
  `ScratchLease(T) { pool, slab, items: []T }` — borrowed pooled scratch for
  non-DType block types (the packed quantized-LHS layouts of [§10](10-quantization.md)). Call
  `lease.release()` exactly once; `items` is valid until then. Release may
  run on any thread.
- **Thread safety.** Both arms are mutex-guarded; buffer release hooks run
  on whatever thread drops the last reference. See [§6.9](06-the-execution-runtime-execcontext-and-the-memory-model.md#69-the-thread-safety-contract).
- **What never enters the pool:** `fromBorrowed*` wraps, load-time weight
  packs, and backend-tier LHS-quantization scratch below the exec seam (a
  deliberate, documented exception).
- **Teardown retention.** Session-lifetime typed buffers (KV-cache f16
  layers, resident bf16 weights) are pool-backed, so tearing down a model
  *session* while keeping the context alive returns their slabs to the free
  list — retained up to the cap so the next session reuses warm slabs.
  `ExecContext.deinit` frees everything.

## 6.6 The worker team (`src/thread.zig`, `src/parallel.zig`)

CPU kernels parallelize over a persistent fork-join team owned by the
context. Everything is lazy: `tryWorkPool` creates the
`thread.Pool` on first request (with `cpuThreadCount(vector_max_threads) - 1`
workers, so the dispatching thread itself is participant 0) and publishes
it in `parallel_pool` (an atomic store with release ordering); the pool in
turn spawns its worker threads only on the first parallel dispatch. A
context that only ever runs small/serial ops never starts a thread. Every
pool-taking kernel call reads the published team through `pc()`, a
`ParallelConfig{ .pool = ... }` snapshot taken per call (acquire load), so
a kernel dispatched from another thread (dot-backward's `OneShotWorker`)
while a lazy `tryWorkPool` retry publishes the pool also sees `Pool.init`'s
writes.

A `Pool.init` failure is latched (`work_pool_failed`): the context logs one
warning and every later `tryWorkPool` returns `error.WorkPoolUnavailable`
without retrying, so `workPool()` stays null and that context runs its
kernels serially for the rest of its life.

```zig
pub fn tryWorkPool(self: *ExecContext) !*thread.Pool   // creates on first call; latched on failure
pub fn workPool(self: *ExecContext) ?*thread.Pool      // tryWorkPool catch null
pub fn pc(self: *const ExecContext) backend.ParallelConfig // { .pool = parallel_pool.load(.acquire) }
pub fn dotBackwardWorker(self: *ExecContext) ?*thread.OneShotWorker
```

The team (`BarrierPool` in `src/thread.zig`) is spin-then-park: after each
dispatch a worker spins on a generation counter for a bounded budget
(default 32768 `spinLoopHint` iterations — the measured M1 tuning, which
survived an x86 sweep), then parks on a futex, so a dense op stream (a
transformer forward) pays atomics instead of kernel round-trips while a
long-idle team consumes no CPU. The dispatcher runs chunk 0 of every
parallel op and spins on the completion counter for a bounded window
(~1M iterations, ≈10 ms on M1-class cores — beyond the straggler tail of a
healthy join, where it therefore behaves as a pure spin and owns a core
until the join), then falls back to 1 ms timed futex waits on the counter
itself — workers never wake that futex, so their completion path is
untouched. The fallback engages whenever the join tail outlives the spin
window — descheduled workers (oversubscription, background load, container
CPU quotas) or a legitimately long final chunk — costs at most 1 ms of
completion-pickup latency per engagement, and after the first wait the spin
window drops to the small bound so the dispatcher stays parked between
re-checks instead of re-burning the full window. Teams sized **above the
physical-core count** (via the `setMaxThreads` oversubscription escape
hatch, or legitimately on a 1-physical-core host where the minimum team of
two exceeds the single core) additionally default the worker spin budget
to 0 — park immediately — and start the dispatcher's spin window at ~4096:
spinning while oversubscribed steals exactly the cores the descheduled
participants need (measured collapse: 19s → 43s prefill on an
HT-oversubscribed i9-13950HX). An explicit, in-range `FUCINA_SPIN_BUDGET`
overrides the worker half of the guard (out-of-range values are treated as
unset); the join fallback is never disabled. On
macOS, workers and dispatcher pin to performance-core QoS
(`pthread_set_qos_class_self_np`); elsewhere the pin compiles to nothing.
`dotBackwardWorker` is a single lazily-started `OneShotWorker` used to
overlap the two branches of matmul backward on native-BLAS builds ([§5](05-automatic-differentiation.md), [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)).

Thread-count knobs, in precedence order:

| Knob | Kind | Effect |
|---|---|---|
| `-Dmax-threads=N` (1–64, default 8 = the M1 Max P-core count) | build option | comptime ceiling for the team and stack task arrays, AND the runtime default team size (`fucina.parallel.vector_max_threads`). Servers with more cores must raise it at build time |
| `fucina.parallel.setMaxThreads(n)` | runtime API | **replaces** the detected CPU count (mirrors llama.cpp `-t`) — it can also *raise* the team size above the detected count, up to the `-Dmax-threads` build ceiling (a team above the physical-core count engages the oversubscription spin guard: worker spin budget → 0); call once at startup before any parallel work; `n == 0` ignored; wins over the env var by pre-seeding the cache |
| `FUCINA_MAX_THREADS` | env var | read once on the first `cpuThreadCount` call; applied as `@min` against the detected count, so it **only lowers**; `0`/invalid = no override |
| `FUCINA_SPIN_BUDGET` | env var | overrides the spin-then-park window, read once per team init; `0` is valid (park immediately — the manual escape for oversubscribed teams, and the automatic default when the team exceeds the physical-core count); workload-coupled and U-shaped — override only with measurements (short budgets, ~512, favor encode-style workloads with serial host sections; the default favors dense LLM op streams) |
| `FUCINA_POOL_PROFILE=1` | env flag | allocates one trace slot per participant and prints fork-join claim/completion timing after every dispatch; diagnostic runs only |

The effective thread count is `parallel.cpuThreadCount(vector_max_threads)`
= `max(1, min(count, vector_max_threads))`, where `count` is the
`setMaxThreads` value verbatim when one was set (detection is bypassed),
else the detected CPU count — clamped to the physical-core count on SMT
machines, so hyperthreads are never double-booked, and further to the
performance-core count on heterogeneous Apple Silicon (with E-cores
enrolled, every fork-join barrier waits on the E-core stragglers; the QoS
pin biases workers to P-cores but cannot guarantee placement — sizing the
team to them does); `setMaxThreads` remains the escape hatch for
deliberate oversubscription — lowered by `FUCINA_MAX_THREADS`. No single
value wins every workload (measured: prefill fastest at all P-cores when
cool, decode often faster one or two threads lower), hence the runtime
knobs. The env parsers behind these knobs are themselves public:
`parallel.envPositiveUsize(name)` implements the positive-usize knob
contract (libc `getenv`, or a libc-free `/proc/self/environ` scan on static
Linux; unset/invalid/`0` ⇒ `null`), `parallel.envNonNegativeUsize(name)` is
the same read with `0` as a value, not "unset" (the `FUCINA_SPIN_BUDGET`
and GPU-floor contract), and `parallel.envFlagValue(name)` is the tri-state
boolean read the tuning table's gates go through (unset/empty ⇒ `null`). `parallel.physicalCpuCount()` (macOS sysctl /
Linux sysfs-topology-over-affinity; null where unknown; probed once and
process-cached, first caller's affinity mask wins) is public as well — it
is the count the oversubscription guard compares the team size against
(deliberately the all-physical-cores reference, so a team sized between
the P-core and all-cores counts is not treated as oversubscribed).
`parallel.performanceCpuCount()` (Apple Silicon sysctl
`hw.perflevel0.physicalcpu`; null elsewhere; same probe-once caching) is
the count the team-size default clamps to on those machines.

```zig
test "worker-team sizing knobs" {
    // Comptime team ceiling from -Dmax-threads (1-64, default 8).
    try std.testing.expect(fucina.parallel.vector_max_threads >= 1);
    // Runtime cap (mirrors llama.cpp -t and FUCINA_MAX_THREADS); call once
    // at startup, before any parallel work.
    fucina.parallel.setMaxThreads(2);
    const n = fucina.parallel.cpuThreadCount(fucina.parallel.vector_max_threads);
    try std.testing.expect(n >= 1 and n <= 2);
}
```

For direct use of the pool (custom parallel sections):

```zig
pub const Pool = struct {
    pub fn init(self: *Pool, options: InitOptions) !void;   // .allocator, .max_workers
    pub fn deinit(self: *Pool) void;
    pub fn parallelChunks(self: *Pool, comptime Task: type,
        tasks: []const Task, comptime run: fn (*const Task) void) void;
    pub fn parallelChained(self: *Pool, comptime Task: type, tasks: []Task,
        initial_count: usize, comptime run: fn (*Task, *const Chain) void) bool;
    pub fn spawnWg / trySpawnWg / waitAndWork;               // std.Io-executor tasks
};
```

`parallelChunks` is the default substrate for splitting a numeric kernel:
fork-join over the hot team, the caller executing chunk 0 and the team the
rest, rendezvousing before return. Degradation is always safe: no barrier,
zero workers, or a team already mid-dispatch (`parallel_chunks_active`)
runs the tasks serially on the caller. `parallelChained` is
dependency-chained fork-join: `tasks[0..initial_count)` start runnable and
a running task makes successors runnable via `chain.enqueue(i)`; it returns
`false` when the team is unavailable or busy (the caller must run the graph
itself). The enqueue contract is strict: across one dispatch, every index
in `[0, tasks.len)` must become runnable **exactly once** (seeds plus
enqueues). Debug/ReleaseSafe builds instrument the contract and panic on
double-enqueue or a stalled under-enqueue; ReleaseFast compiles the checks
out, where a violation corrupts the intrusive Treiber stack (the pop is
ABA-unsafe) or spins forever. `spawnWg` / `trySpawnWg` / `waitAndWork`
route general async tasks through `std.Io`'s executor with a
`thread.WaitGroup` — unlike the hot team, each spawn heap-allocates a task
node and parks/wakes via futex syscalls. Per-kernel threading policy (work
thresholds such as `parallel.vector_matmul_work_threshold`, claim-chunk
sizing) and the backend-side pool handshake are [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md) material ([§9.4](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#94-native-backend-portable-vector-kernels-srcbackendvector), [§9.8](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#98-threading-the-worker-team-srcthreadzig-srcparallelzig)).
`ParallelConfig` is an internal backend/vector type, not part of the public
surface.

With `FUCINA_POOL_PROFILE=1`, each participant writes only its own trace slot,
so claim timing adds no trace atomics. The dispatcher's acquire join makes the
slots visible before it prints offsets relative to dispatch start. A zero-task
participant is still reported: its completion offset is the barrier-tail
evidence needed to distinguish late wake-up from slow task execution.

## 6.7 RhsLifetime: address-keyed caching of RHS operands (`src/exec/quant_matmul.zig`)

```zig
pub const RhsLifetime = enum {
    transient,       // default: no address-keyed caching beyond this dispatch
    stable_process,  // caller guarantees stable bytes; backends may cache wraps
    pub fn isCacheable(self: RhsLifetime) bool // true iff .stable_process
};
```

Re-exported as `fucina.RhsLifetime`. GPU backends ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)) avoid re-wrapping and
re-uploading a quantized weight on every matmul by caching device wraps
keyed on the RHS byte address. That is only sound if the bytes at that
address never change, which the type states explicitly:

- `.transient` — ordinary tensor/temporary storage. The backend may still
  use the GPU for the dispatch, but must not retain an address-keyed wrap.
- `.stable_process` — the caller guarantees the RHS bytes stay mapped at the
  same address for the process lifetime (an mmap'd weight file kept mapped),
  or are device-resident storage from
  `fucina.internal.gpu.allocResidentBytes` whose owner evicts cached wraps
  via `freeResidentBytes` before freeing. Violating the promise is
  use-after-free on the GPU side.

The hard rule: **pooled storage must never be marked `.stable_process`.**
The slab arm makes address reuse routine, so a cached wrap keyed on a pooled
transient's address would silently read stale data after the slab is
recycled. Every in-tree `.stable_process` caller wraps device-resident or
mmap'd weight bytes (`src/weights.zig` threads the flag through the
quantized-weight loaders onto `QuantMatmul.rhs_lifetime`). This is
about storage stability, not about whether the operand is a model weight.

## 6.8 Determinism and the RNG contract (`src/rng.zig`)

`fucina.rng` is the repo-owned deterministic RNG (splitmix64-based). Its
(seed → values) mappings are **checkpoint contracts**: consumers store a
seed and regenerate values instead of serializing them, so none of these
functions may ever change behavior or depend on `std.Random` internals
(which are free to change across Zig releases).

```zig
pub fn splitmix64(state: *u64) u64                     // one sequential step
pub fn at(seed: u64, i: u64) u64                       // i-th output, O(1), counter-based
pub fn gaussianFill(seed: u64, out: []f32, scale: f32) void
pub fn gaussianFillAt(seed: u64, first: u64, out: []f32, scale: f32) void
pub fn gaussianFillAtFast(seed: u64, first: u64, out: []f32, scale: f32) void
pub fn uniformFill(seed: u64, out: []f32, lo: f32, hi: f32) void
pub fn kaimingUniformFill(seed: u64, out: []f32, fan_in: usize) void
pub fn normalFill(seed: u64, out: []f32, mean: f32, std_dev: f32) void
pub fn gumbelFill(seed: u64, out: []f32) void
pub fn randintFill(seed: u64, out: []i64, low: i64, high: i64) void
pub fn randpermFill(seed: u64, out: []i64) void
```

- `at(seed, i)` computes the i-th output of the stream started at `seed`
  directly — every element is a pure function of `(seed, i)`, independent of
  the preceding ones.
- `gaussianFill` is splitmix64 + Box-Muller (two stream outputs per value
  pair); `gaussianFillAt` is its counter-based form: filling elements
  `first .. first + out.len` of the same stream, bitwise identical to the
  sequential fill under **any** range decomposition.
- `gaussianFillAtFast` is a vectorized variant with f32 polynomial
  transcendentals. It is a **distinct** (seed → values) mapping — values
  agree with `gaussianFillAt` to a few ulps but are not bitwise equal — and
  therefore a separate checkpoint contract. It is equally
  chunking-invariant.
- `uniformFill` maps one output per value onto `[lo, hi)` (half-open bound
  kept exact by clamping the rare round-up); `kaimingUniformFill` is the
  PyTorch `nn.Linear`/LoRA-A default init; `normalFill` adds explicit
  moments on top of `gaussianFill`.
- `gumbelFill` is standard Gumbel(0, 1) by inverse CDF over a strictly
  OPEN uniform (`u = ((x >> 11) + 0.5)·2^-53`, so `-ln(-ln(u))` is always
  finite); `randintFill` maps one output per value onto `[low, high)` by
  the widening multiply-shift `low + ((x·span) >> 64)` (bias below
  `span·2^-64`, full-i64 spans handled in two's-complement);
  `randpermFill` is Fisher–Yates over the counter stream (step k swaps
  with `(at(seed, n-1-k)·(k+1)) >> 64`). All three share the checkpoint
  contract.

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

Where the contract is load-bearing:

- **Dropout** ([§4](04-tensor-operations.md), [§5](05-automatic-differentiation.md)): the mask is never stored — forward, backward, and
  `checkpoint` recompute all regenerate it from `(seed, element index)` via
  `at`, so the op is a deterministic pure function of `(input, p, seed)` and
  parallel kernels are bitwise-stable regardless of chunking. Pass a fresh
  seed per call (reusing a seed reuses the mask); eval mode is simply not
  calling dropout.
- **APOLLO** ([§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md), `src/optim.zig`): low-rank projections are regenerated
  from their stored seed at checkpoint restore via `gaussianFill`, not
  serialized.
- **Evolution strategies** ([§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md), `src/es.zig`): member perturbations are
  regenerated from member seeds via `gaussianFillAtFast` in parallel chunks,
  with results independent of the chunking.

These are the determinism guarantees the runtime makes: seed-driven ops are
bitwise reproducible across runs, thread counts, and chunk decompositions.
No blanket bitwise-reproducibility claim is made for every parallel
reduction at every thread count; where a kernel guarantees serial/parallel
parity, the guarantee is pinned by its tests ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)).

Every guarantee above is stated under the default IEEE floating-point
environment. That environment is per-thread state this process shares with
the linked CBLAS and the GPU driver, so it is checkable rather than assumed:
see [§6.10](06-the-execution-runtime-execcontext-and-the-memory-model.md#610-the-ieee-floating-point-environment-srcfpenvzig).

## 6.9 The thread-safety contract

What is thread-safe inside a context:

- **The allocator**: `init` wraps the caller's allocator in
  `thread.ThreadSafeAllocator` (a mutex around alloc/resize/remap/free), so
  internal allocations and frees may happen on worker threads.
  `ctx.allocator()` returns this wrapper.
- **The BufferPool**: both arms are mutex-guarded, `outstanding` is atomic,
  and buffer/slab release hooks run on whatever thread drops the last
  reference (storage refcounts are atomic).
- **Lazy initialization**: `tryWorkPool` and `dotBackwardWorker` are
  mutex-guarded and idempotent; a failed pool init is latched, not retried.

What is not:

- **Op execution, scope open/close, and every other context mutation are
  single-threaded**: drive one `ExecContext` from one thread at a time. CPU
  ops fan work out to the team and join before returning. The backward
  engine is the documented exception: it may run VJPs, and checkpoint
  recomputes, on pool threads, and a recompute installs a scope stack of
  its own for its thread ([§6.3](06-the-execution-runtime-execcontext-and-the-memory-model.md#63-exec-scopes-implicit-ownership-for-training-srcexeczig-srcexecruntimezig)) so the context's stack stays
  single-threaded. Eligible f32/f16/dense-quant
  GPU ops submit before return and keep program order through their provider
  queue/stream; a later CPU access performs the storage readiness wait. The
  external call order remains serial.
- **Sharing tensor handles across threads is unspecified.** The runtime
  makes no promise about concurrent reads or writes through tensor handles
  on different threads (storage refcounts are atomic, but the handle structs
  are mutable value types with interior pointers). Confine a tensor and its
  views to the thread driving its context, or synchronize externally.
  CPU parallelism inside the runtime — kernel chunking, ES perturbation fills,
  dot-backward branches — is always mediated by the context's own team and
  joins before the op returns. Submitted GPU completion is the explicitly
  documented exception (`GPU-OFFLOAD.md`).

## 6.10 The IEEE floating-point environment (`src/fpenv.zig`)

Every numeric contract in this library — the RNG's (seed → values) mapping
([§6.8](06-the-execution-runtime-execcontext-and-the-memory-model.md#68-determinism-and-the-rng-contract-srcrngzig)), scalar-vs-native backend parity ([§9.3](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#93-the-scalar-reference-arms-and-the-parity-contract-srcbackendvector-srcbackendparity_testzig)), thread-count-invariant
kernels ([§9.4](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#94-native-backend-portable-vector-kernels-srcbackendvector)) — is stated under the **default IEEE environment**: round to
nearest-even, gradual underflow, no trapping. Nothing in the language
guarantees it. The rounding mode and the flush-to-zero bit live in a
per-thread control register (`FPCR` on aarch64, `MXCSR` on x86_64) shared
with everything else on the thread, including the external CBLAS and the GPU
driver. A vendor kernel that enables flush-to-zero and does not restore it
changes the numerics of every op that follows, silently: results stay
plausible and stop matching the oracles.

`fucina.fpenv` makes that state observable, controllable within a scope, and
assertable. It is inquiry-first in the Fortran `ieee_arithmetic` sense —
`supported` reports whether the target exposes the registers, the getters
return `null` where it does not, and the setters are no-ops there — so
calling code never has to branch on the architecture. Nothing here sits on an
op hot path.

```zig
pub const supported: bool                     // aarch64 / x86_64 today
pub const RoundingMode = enum { nearest_even, toward_zero, upward, downward };
pub const UnderflowMode = enum { gradual, flush_to_zero };
pub const Environment = struct { rounding: RoundingMode, underflow: UnderflowMode,
                                 pub fn isDefault(self) bool };
pub const default_environment: Environment    // nearest_even + gradual

pub fn get() ?Environment
pub fn set(env: Environment) void
pub fn roundingMode() ?RoundingMode
pub fn underflowMode() ?UnderflowMode
pub fn assertDefault() error{NonDefaultFloatEnvironment}!void

pub const Exceptions = struct { invalid, division_by_zero, overflow, underflow,
                                inexact, denormal: bool,
                                pub fn any(self) bool
                                pub fn anySignificant(self) bool      // any but inexact
                                pub fn unionWith(self, other) Exceptions };
pub fn testExceptions() Exceptions
pub fn clearExceptions() void
pub fn raiseExceptions(flags: Exceptions) void

pub const Guard = struct { pub fn begin() Guard; pub fn restore(self) void };
pub const Probe = struct { pub fn begin() Probe; pub fn sample(self) Exceptions;
                           pub fn end(self) Exceptions };
```

- `Guard` is Fortran's rule that a procedure changing the environment restores
  it before returning, made explicit: `begin` snapshots, `restore` puts it
  back, and `restore` is idempotent and safe on unsupported targets.
- `Probe` measures which IEEE exceptions a region raised. `begin` takes over
  the accrued flags (saving and clearing them), `sample` reads what has
  accrued so far, and `end` reads the final set and merges the saved outer
  flags back in, so probes nest without an inner region erasing an outer
  one's history. Flags are per-thread: a probe around a parallel op observes
  only the calling thread's, not the worker team's.
- `Exceptions.denormal` is not one of the five IEEE exceptions. Both
  supported architectures expose a subnormal-operand flag (x86 `DE`, aarch64
  `IDC`) and it is informative when auditing quantization block scales, so it
  is surfaced alongside them. `anySignificant()` is the "worth looking at"
  predicate: everything except `inexact`, which ordinary arithmetic raises
  constantly.

On the context side, `ExecContext` records the environment at `init` and
compares on demand:

```zig
pub fn checkFloatEnvironment(self: *const ExecContext) !void   // error.FloatEnvironmentChanged
pub fn floatEnvironmentAtInit(self: *const ExecContext) ?Environment
```

Call `checkFloatEnvironment` after crossing into foreign code, or around a run
whose output is compared bitwise. Both succeed unconditionally where the
target does not expose the environment: the facility never reports a wrong
answer in place of "cannot observe". The in-tree gate that the GEMM dispatch
tier (BLAS included) leaves the environment intact lives in
`src/exec_tests.zig`.

```zig
test "float environment: assert the default, scope a change, probe exceptions" {
    const fpenv = fucina.fpenv;
    if (!fpenv.supported) return; // nothing to observe on this target

    // The contract every pinned numeric result in this library assumes.
    try fpenv.assertDefault();

    // A scoped change: the guard puts the environment back.
    {
        var guard = fpenv.Guard.begin();
        defer guard.restore();
        fpenv.set(.{ .rounding = .toward_zero, .underflow = .gradual });
        try std.testing.expectEqual(fpenv.RoundingMode.toward_zero, fpenv.roundingMode().?);
    }
    try fpenv.assertDefault();

    // Which exceptions did this region raise?
    var probe = fpenv.Probe.begin();
    var huge: f32 = 3.0e38;
    std.mem.doNotOptimizeAway(&huge);
    var squared = huge * huge;
    std.mem.doNotOptimizeAway(&squared);
    const raised = probe.end();
    try std.testing.expect(raised.overflow);
    try std.testing.expect(raised.anySignificant());
    fpenv.clearExceptions();

    // A context checks its own environment against creation time.
    var ctx: fucina.ExecContext = undefined;
    ctx.init(std.testing.allocator);
    defer ctx.deinit();
    try ctx.checkFloatEnvironment();
}
```
