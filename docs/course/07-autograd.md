# Chapter 07 — Autograd: the graph hidden in the values

*Part III — Learning*

Everything we have built so far computes in one direction. [Chapter 3](03-tensors-from-scratch.md)
gave us tensors, [Chapter 5](05-the-operation-library.md) an eager operation
library, [Chapter 6](06-going-fast-on-cpus.md) made it fast. But to *learn*,
a model needs the loss to flow backward: for every parameter, "how much
would the loss change if I nudged this number?" Computing that gradient by
hand for a real network is out of the question; computing it mechanically,
for *any* composition of ops, is the job of automatic differentiation.

This chapter builds that machinery twice: first a complete scalar autograd
in about sixty lines of fresh Zig — small enough to hold in your head, and
including the one genuinely subtle bug every implementer hits — then the
real engine in `src/ag/`, the same idea at tensor rank with ownership,
concurrency, and verification done properly. The design has an unusual
signature to watch for throughout: **there is no graph object**. The live
tensors *are* the graph.

## 7.1 What backward() must compute

Recall the setup from [Chapter 2](02-just-enough-ml.md): a model is a
parameterized function, a loss measures how wrong its output is, and
training nudges every parameter against its partial derivative `∂L/∂θ`. The
vector of all those partials is the gradient, and the chain rule says we can
get it by composing *local* derivatives along the path from each parameter
to the loss.

The insight that makes this practical is **reverse-mode** differentiation.
A loss is one scalar; a model has millions of parameters. Forward mode would
propagate "sensitivity to θᵢ" through the network once *per parameter* —
millions of sweeps. Reverse mode runs a single backward sweep from the loss
and recovers the sensitivity of that one output to *everything* along the
way: one forward pass, one backward pass, all gradients.

> **ML note** — The two modes are transposes: forward mode answers "if I
> wiggle this input, what happens to all outputs?" (cheap when inputs are
> few); reverse mode answers "to move this output, how should all inputs
> wiggle?" (cheap when outputs are few). Deep learning has one scalar output
> and a mountain of inputs, so reverse mode wins by the width of the mountain.

The unit of work in reverse mode is the **vector–Jacobian product**, or VJP.
An op never sees the whole network; it knows exactly one local rule:

> given the gradient of the loss with respect to my *output* (call it `gy`),
> produce the gradient of the loss with respect to each of my *inputs*.

For `y = relu(x)` that rule is one line — the gradient passes where the
input was positive and dies where it was not — and here it is in the real
engine, from `src/ag/backward.zig`:

```zig
dst.* = if (value > 0) grad else 0;
```

That single line is the entire calculus of ReLU. For `z = x * y` it is the
product rule: `gx = gy * y`, and symmetrically. No op ever materializes a
Jacobian matrix; each hands back gradients the same shape as its inputs, and
chaining the rules backward from the loss *is* the chain rule, executed as
data flow.

Two questions remain, and they are the whole engineering content of an
autograd engine: **topology** — when `backward()` runs, how does it know
which ops happened, feeding what? — and **scheduling** — a value consumed by
several branches must receive the *sum* of all downstream contributions
before it propagates further; who ensures that, and exactly once?

## 7.2 Where is the graph?

Most frameworks answer the topology question with an explicit data
structure: record every op onto a **tape** and replay it in reverse; or
build a **graph object** and hand it to an engine; or **trace** the program
once and differentiate the trace. All of these introduce a thing that exists
apart from your values — a registry with its own lifetime, its own
invalidation rules, its own API.

Fucina's answer comes from its ancestry. Quoting the opening paragraph of
the README's *Origins* section (`README.md`, "Origins") in full:

> Fucina grew out of autograd concepts I first explored in Go with
> [spaGO](https://github.com/nlpodyssey/spago) — above all the idea that the
> graph should be **implicit in the values themselves**: no graph object, no
> tape, no persistent engine. Each result carries a pointer to the operation
> that produced it, and `backward()` discovers the topology by walking those
> pointers. spaGO executed that idea the Go way: one goroutine per node,
> each blocking until its gradient contributions arrived, the runtime
> scheduler absorbing the wait. Zig has no goroutines, so Fucina keeps the
> idea and rethinks the execution: every node carries an atomic dependency
> counter, and its gradient fires only when the counter drains — concurrent,
> on a bounded worker pool, no blocked workers (`src/ag/`). (AFAIK) Mainstream
> stacks route backward through a central engine over an explicit node graph
> or a trace; here the live tensors *are* the graph.

Unpack the claim. Every op result may carry a pointer to a record of "the op
that made me", and that record holds pointers to the *operand* results.
Follow the pointers from the loss and you recover the entire computation.
There is nothing else: no registry to clear, no session to reset. Deinit the
tensors and the graph is gone, because it never was anything but the
tensors.

This is consistent with the library's broader stance — "Fucina is
deliberately **eager and local**: no global graph object, no fusion pass, no
compiler layer" (`README.md`, Design) — and, unusually, the simplicity is
*enforced*: a guard test in `src/ag_tests.zig` asserts that the legacy
`Function`/`Node`/`Engine` declarations of an earlier, heavier design stay
deleted from the core (`docs/REFERENCE.md:3590-3595`).

The scheduling question — spaGO's goroutines versus Fucina's atomic counters —
we will meet twice: once in miniature in the next section, and once for real
in §7.7.

## 7.3 Build it yourself: a scalar autograd

Before touching `src/ag/`, let's build the idea at rank zero — plain `f32`
scalars — where nothing can hide. *This is course code, not repo code*; both
versions below compile and pass under `zig test` with Zig 0.16.0.

A differentiable value needs three things: its data, a slot for the gradient
it will accumulate, and — if an op produced it — pointers to the operands.
That last field is the whole "implicit graph" idea:

```zig
const Value = struct {
    data: f32,
    grad: f32 = 0,
    rule: Rule = .leaf,

    const Rule = union(enum) {
        leaf,
        add: [2]*Value,
        mul: [2]*Value,
    };
};

fn add(a: *Value, b: *Value) Value {
    return .{ .data = a.data + b.data, .rule = .{ .add = .{ a, b } } };
}

fn mul(a: *Value, b: *Value) Value {
    return .{ .data = a.data * b.data, .rule = .{ .mul = .{ a, b } } };
}
```

> **Zig note** — `Rule` is a *tagged union*: a value is exactly one of
> `.leaf`, `.add`, or `.mul`, and the payload (here a two-element array of
> pointers) is only accessible after you `switch` on the tag. It is Zig's
> type-safe answer to C's `union` + discriminant-by-convention, and the
> natural encoding for "which op made me". Note also that the graph here
> lives entirely on the stack: `mul(&x, &y)` returns a `Value` holding
> pointers to your locals. No allocator in sight — yet.

### The naive walk, and where it breaks

The obvious `backward()` — two methods inside `Value` — seeds the output
with `dL/dL = 1` and recursively pushes gradient into parents:

```zig
    fn backward(self: *Value) void {
        self.grad = 1; // seed: dL/dL = 1
        self.propagate();
    }

    // BUG: propagates into a parent as soon as one contribution arrives.
    fn propagate(self: *Value) void {
        switch (self.rule) {
            .leaf => {},
            .add => |ps| {
                ps[0].grad += self.grad;
                ps[0].propagate();
                ps[1].grad += self.grad;
                ps[1].propagate();
            },
            .mul => |ps| {
                ps[0].grad += self.grad * ps[1].data;
                ps[0].propagate();
                ps[1].grad += self.grad * ps[0].data;
                ps[1].propagate();
            },
        }
    }
```

For chains and trees this is correct: `z = x * y` with `x = 3, y = 5` gives
`x.grad == 5` and `y.grad == 3` — the product rule, our first passing test.

Now share a node. Let `s = x + 1` and `y = s * s`, with `x = 2`, so `s = 3`
and `y = s² = 9`. Calculus says `dy/dx = 2s = 6`. Trace the naive walk:

1. Seed: `y.grad = 1`. `y` is a `mul` whose operands are both `s`.
2. First operand: `s.grad += 1 * 3` → `s.grad = 3`. Recurse: the `add` pushes
   `3` into `x`. So far `x.grad = 3`.
3. Second operand: `s.grad += 1 * 3` → `s.grad = 6`. Recurse *again*: the
   `add` pushes the **entire current** `s.grad` into `x` a second time.
   `x.grad = 3 + 6 = 9`.

Nine, not six: the second walk through `s` re-delivered the first
contribution on top of the second. The failure isn't exotic — any weight
used twice, any residual connection, any shared activation builds this
diamond. The bug is pinned by a compile-checked test that deliberately
*asserts the wrong value*, so the failure mode is on record:

```zig
test "DAGs break: a shared interior node is walked twice" {
    var x = Value{ .data = 2 };
    var one = Value{ .data = 1 };
    var s = add(&x, &one); // s = 3, shared below
    var y = mul(&s, &s); //   y = s^2, so dy/dx should be 2s = 6
    y.backward();
    try std.testing.expectEqual(@as(f32, 9), x.grad); // 3s, NOT 2s. Wrong.
}
```

The lesson: **a node must fire only after ALL of its consumers have
delivered their contributions**, and then exactly once. Recursion order
cannot guarantee that on a DAG. We need to *count*.

### The fix: a pending counter

Split backward into two passes. Pass one walks the parent pointers from the
output and counts, on every node, how many gradient contributions it should
expect — one per consumer edge. Pass two delivers contributions; a node
accumulates silently until its counter drains to zero, and only then fires
its local rule, propagating its now-complete gradient exactly once.

Here is the complete fixed engine — this is the centerpiece of the chapter,
and it is the real design in miniature:

```zig
// COURSE CODE — a complete scalar autograd in ~60 lines. The graph is
// implicit: each op result holds pointers to the values that produced it.
// No tape, no engine object. Fixed against the DAG bug with a pending
// counter, exactly the scheme Fucina's src/ag/core.zig uses at tensor rank.
const std = @import("std");

const Value = struct {
    data: f32,
    grad: f32 = 0,
    rule: Rule = .leaf,
    pending: u32 = 0, // consumer edges not yet delivered this pass

    const Rule = union(enum) {
        leaf,
        add: [2]*Value,
        mul: [2]*Value,
    };

    /// Pass 1 — discover the topology by walking parent pointers.
    /// Each consumer edge adds one pending contribution; recurse into a
    /// node's parents only on first visit.
    fn prepare(self: *Value) void {
        self.pending += 1;
        if (self.pending > 1) return; // already visited this pass
        switch (self.rule) {
            .leaf => {},
            .add, .mul => |ps| {
                ps[0].prepare();
                ps[1].prepare();
            },
        }
    }

    /// Pass 2 — accumulate a contribution; fire the local rule only when
    /// the LAST expected contribution has landed.
    fn contribute(self: *Value, g: f32) void {
        self.grad += g;
        self.pending -= 1;
        if (self.pending > 0) return; // more consumers still owe us gradient
        switch (self.rule) {
            .leaf => {},
            .add => |ps| {
                ps[0].contribute(self.grad);
                ps[1].contribute(self.grad);
            },
            .mul => |ps| {
                ps[0].contribute(self.grad * ps[1].data);
                ps[1].contribute(self.grad * ps[0].data);
            },
        }
    }

    fn backward(self: *Value) void {
        self.prepare(); //     the seed counts as one pending contribution
        self.contribute(1); // dL/dL = 1
    }
};

fn add(a: *Value, b: *Value) Value {
    return .{ .data = a.data + b.data, .rule = .{ .add = .{ a, b } } };
}

fn mul(a: *Value, b: *Value) Value {
    return .{ .data = a.data * b.data, .rule = .{ .mul = .{ a, b } } };
}
```

> **Zig note** — `.add, .mul => |ps|` is a multi-prong switch capture: both
> variants carry the same payload type (`[2]*Value`), so one arm handles
> both. And notice `prepare` doubles as the visited-set: `pending > 1` on
> entry means "already counted this pass", so the recursion into parents
> happens exactly once per node — no separate hash set, the counter *is* the
> bookkeeping.

The tests now assert the *right* values, including the diamond that broke the
naive version and the fan-out identity Fucina's own documentation uses:

```zig
test "shared interior node fires exactly once" {
    var x = Value{ .data = 2 };
    var one = Value{ .data = 1 };
    var s = add(&x, &one); // s = 3, consumed twice below
    var y = mul(&s, &s); //   y = s^2
    y.backward();
    try std.testing.expectEqual(@as(f32, 6), x.grad); // 2s. Correct.
}

test "fan-out accumulates: d/dx (x^2 + 4x) = 2x + 4" {
    var x = Value{ .data = 3 };
    var four = Value{ .data = 4 };
    var sq = mul(&x, &x); // x feeds this node twice
    var lin = mul(&four, &x); // and a third consumer here
    var loss = add(&sq, &lin);
    loss.backward();
    try std.testing.expectEqual(@as(f32, 10), x.grad); // 2*3 + 4
}
```

> **ML note** — "fan-out accumulates" is not an implementation detail; it is
> the mathematics of weight sharing. A weight used in two places contributes
> to the loss through both, and its gradient is the sum of both paths. Every
> shared embedding, every residual stream, every reused projection relies on
> the property this counter enforces.

### From the toy to the real engine

Everything in the toy has a named counterpart in `src/ag/`:

| Toy (course code) | Fucina (`src/ag/`) |
| --- | --- |
| `Value` with `grad` + `pending` | `GradState` (`core.zig:100`) |
| `rule: Rule` tagged union | `grad_fn: ?BackwardFunction` — a type-erased VJP record |
| `.leaf` variant | `grad_fn == null` (leaf `GradState`) |
| `prepare()` | `prepareBackwardPass` (`core.zig:173-184`) |
| `pending -= 1; fire at zero` | `pending_grads.fetchSub(1, .acq_rel)` draining (`core.zig:273-277`) |
| `contribute` recursion | ready-node scheduling on a bounded worker pool |
| `grad += g` | mutex-guarded accumulation with copy-on-write |
| (absent) | ownership, constants vs variables, seeding rules, error atomicity, one-backward-per-graph |

The last row is the honest one: no allocator (the toy's graph borrows your
stack frame), no constants, no protection against running backward twice,
and happy single-threaded recursion. The rest of this chapter is what it
takes to do each of those properly.

## 7.4 The real node: GradState and the VJP records

The engine lives in `src/ag/core.zig` — 733 lines *including* two in-file
tests, as counted today with `wc -l`; the siblings are similarly small
(`control.zig` 75, `custom.zig` 227, `gradcheck.zig` 194, `checkpoint.zig`
435). The bulk of `src/ag/` is `backward.zig` at 5420 lines — but that is
the *inventory* of per-op VJP records, each mechanical. The machinery they
plug into is one struct, one interface, one scheduler.

The struct — the entire per-node engine state (`src/ag/core.zig:100-115`):

```zig
pub const GradState = struct {
    allocator: Allocator,
    grad: ?Tensor = null,
    grad_fn: ?BackwardFunction = null,
    state: std.atomic.Value(u8) = std.atomic.Value(u8).init(@intFromEnum(BackwardState.idle)),
    pending_grads: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    grad_mutex: thread.Mutex = .{},
    /// Set once a backward pass with this state as an OUTPUT completes.
    /// The pass leaves its gradient contributions accumulated in every
    /// interior state of the graph, so a second pass over the same graph
    /// would compound them; `backwardGradImpl` rejects a marked output with
    /// `AgError.BackwardAlreadyRun` before installing any scheduling state.
    /// Leaves (`grad_fn == null`) are never marked — they have no graph to
    /// consume. Touched only on the thread driving the pass, never from
    /// pool tasks, so it needs no synchronization.
    backward_done: bool = false,
```

You can read the toy straight through it: `grad` is the accumulator,
`pending_grads` the counter (atomic now — contributions may arrive from pool
threads), `grad_fn` the tagged union generalized. A **leaf** — a trainable
parameter — is a `GradState` with `grad_fn == null`; an op result is one
with a VJP record attached. Nothing else exists.

One note before we look inside: `GradState`, `BackwardFunction`, and the
`backwardGrad*` entry points are engine internals — deliberately *not*
re-exported at the `fucina` root (`docs/REFERENCE.md:3599-3601`). Your code
never names them; you meet them here because we are reading the engine.

The interface — `grad_fn` type-erased so hundreds of different op records fit
one scheduler (`src/ag/core.zig:23-33`):

```zig
pub const BackwardFunction = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        operands: *const fn (*const anyopaque) []const ?*GradState,
        backward: *const fn (*const anyopaque, *ExecContext, *const Tensor, []const bool, []?Tensor) anyerror!void,
        deinit: *const fn (*anyopaque, Allocator) void,
        prefer_async_backward: bool = false,
        estimated_work: ?*const fn (*const anyopaque) usize = null,
    };
```

> **Zig note** — Zig has no inheritance and no closures, so runtime
> polymorphism is built by hand: a fat pointer (`ptr: *anyopaque`) plus a
> vtable of function pointers, each record downcasting with
> `@ptrCast(@alignCast(ptr))`. It is what a C++ compiler would generate for
> a virtual class — except you can read it, and nothing is hidden.

`operands()` **is the graph**: one `?*GradState` slot per forward operand,
`null` where the operand was a constant. The two extra vtable fields are the
concurrency opt-in we will meet in §7.7.

### Anatomy of a VJP record

Here is the cleanest record in the inventory, in full
(`src/ag/backward.zig:472-517`):

```zig
pub const ReluBackward = struct {
    parents: [1]?*GradState,
    input: RawTensor,

    pub fn init(self: *ReluBackward, allocator: std.mem.Allocator, parent: ?*GradState, input: *const RawTensor) !void {
        _ = allocator;
        self.* = .{
            .parents = .{parent},
            .input = try input.cloneView(),
        };
    }

    fn operands(ptr: *const anyopaque) []const ?*GradState {
        const self: *const ReluBackward = @ptrCast(@alignCast(ptr));
        return self.parents[0..];
    }

    fn backward(ptr: *const anyopaque, ctx: *ExecContext, gy: *const RawTensor, needs_grad: []const bool, out: []?RawTensor) !void {
        const self: *const ReluBackward = @ptrCast(@alignCast(ptr));
        if (needs_grad.len == 0 or !needs_grad[0]) return;

        var x = try contiguousForRead(ctx, &self.input);
        defer x.deinit();
        var gy_ready = try contiguousForRead(ctx, gy);
        defer gy_ready.deinit();

        var gx = try ctx.empty(x.shape.slice());
        errdefer gx.deinit();
        for (x.dataConst(), gy_ready.dataConst(), gx.data()) |value, grad, *dst| {
            dst.* = if (value > 0) grad else 0;
        }
        out[0] = gx;
    }

    fn deinit(ptr: *anyopaque, allocator: std.mem.Allocator) void {
        const self: *ReluBackward = @ptrCast(@alignCast(ptr));
        self.input.deinit();
        core.destroyNode(ReluBackward, allocator, self);
    }

    pub const vtable = BackwardFunction.VTable{
        .operands = operands,
        .backward = backward,
        .deinit = deinit,
    };
};
```

Every record in the 5420-line inventory has this shape. Three things to
notice:

- **It saves a view, not a copy.** `init` stores `input.cloneView()` — a
  refcounted view of the operand's storage ([Chapter 3](03-tensors-from-scratch.md)).
  Whatever the forward pass later does with its tensors, the view keeps the
  bytes alive — the reason values and graph nodes can have different
  lifetimes (§7.8).
- **It respects `needs_grad`.** A constant operand costs nothing: the engine
  passes `false` and the record computes no gradient for it (§7.6).
- **The math is one line** — the derivative of ReLU is the `if`; everything
  else is plumbing, ending with an owned tensor in `out[0]` that the engine
  consumes.

> **Zig note** — `for (a, b, c) |value, grad, *dst|` zips three slices in one
> loop, with a safety check that they have equal length (a checked panic in
> safe builds) and a pointer capture on the one being written. It is the idiomatic replacement for an
> indexed loop, and the backend ([Chapter 6](06-going-fast-on-cpus.md)) can
> vectorize it just as well.

### One allocation per node

A detail worth savoring: the `GradState` header and the typed record are
co-allocated as a single heap node (`src/ag/core.zig:81-98`):

```zig
pub fn createNode(comptime Record: type, init_args: anytype) !*GradState {
    const allocator: Allocator = init_args[0];
    const node = try allocator.create(BackwardNode(Record));
    errdefer allocator.destroy(node);
    try @call(.auto, Record.init, .{&node.record} ++ init_args);
    node.state = .{
        .allocator = allocator,
        .grad_fn = .{ .ptr = &node.record, .vtable = &Record.vtable },
    };
    return &node.state;
}
```

and the tail of every record's vtable `deinit` is `destroyNode`
(`core.zig:95-98`), which recovers the whole node from the record pointer —
`@fieldParentPtr("record", record)` — and frees it, header included.

> **Zig note** — `BackwardNode(Record)` is a function returning a type
> (comptime generics, [Chapter 4](04-axes-with-names.md)): a struct holding
> `{ state: GradState, record: Record }`. `@fieldParentPtr` recovers the
> containing struct from a pointer to one of its fields — the classic
> `container_of` idiom from kernel C, but checked by the type system. Net
> effect: one allocation per graph node on the hot path, freed in one
> `destroy`.

## 7.5 One funnel: constants, variables, and finishOp

Who gets a `GradState`? The facade (`src/ag/tensor.zig`) draws the line at
construction:

- **Constants** — `fromSlice`, `constant`, `fromTensor`, `zeros`, and
  friends — have `grad_state == null`: they participate in any op but never
  accumulate gradients (inputs, labels, masks, frozen weights).
- **Variables** — `variable`, `variableFromSlice` — get a *leaf* `GradState`
  (`grad_fn == null`): the trainable parameters.
- **Op results** whose operands require gradients get an *interior*
  `GradState` carrying the VJP record.

The query is exactly the null check you'd hope for (`src/ag/tensor.zig:926-928`):

```zig
pub fn requiresGrad(self: *const Self) bool {
    return self.grad_state != null;
}
```

And every differentiable op in the library — all of them — ends in one
private tail (`src/ag/tensor.zig:5673-5690`):

```zig
fn finishOp(
    comptime result_tags: anytype,
    ctx: *ExecContext,
    value: RawTensor,
    wants_grad: bool,
    comptime BackwardType: type,
    create_args: anytype,
) !Tensor(result_tags) {
    if (!wants_grad or !control.isGradEnabled()) return finishNoGrad(result_tags, ctx, value);
    if (ctx.execScopeActive()) try ctx.reserveScopeSlot();
    const state = try core.createNode(BackwardType, create_args);
    var out = try finishWithBackward(result_tags, value, state);
    if (ctx.execScopeActive()) {
        adoptIntoScope(ctx, &out);
        out.scope_owned = true;
    }
    return out;
}
```

Ten lines decide everything. If no operand wants gradients — or a `noGrad`
scope is active (§7.8) — the result is a plain no-grad tensor and *no graph
state is retained*. Otherwise the eagerly computed value is wrapped with a
fresh node and, if an exec scope is open, adopted by it (the scope slot is
reserved *before* the value is consumed, so adoption cannot fail afterwards).

The consequence is stated plainly in the reference and is worth engraving:
"Because forward always takes the identical kernel path, training and
inference produce identical values" (`docs/REFERENCE.md:3642-3648`). Training
mode is not a different execution engine; it is this one branch.

The whole user-facing story fits in one machine-verified snippet —
`test "backward and grad read"` in `docs/REFERENCE.md` §5.1 (like every
`test "..."` block quoted from REFERENCE.md in this chapter, it runs against
the real modules under the in-tree `zig build snippet-check` gate): a
variable `x`, a constant `c`, `loss = sum(x·c)`, one `backward`; `x.grad()`
reads back exactly `c`, while `c.grad()` returns `null`. The gradient API is
small — today's signatures (the API is explicitly unstable; no semver yet),
from `src/ag/tensor.zig`:

```zig
pub fn backward(self: *const Self, ctx: *ExecContext) !void
pub fn backwardWithGrad(self: *const Self, ctx: *ExecContext, grad_output: *const Self) !void
pub fn grad(self: *const Self, ctx: *ExecContext) !?Self     // deep copy
pub fn gradView(self: *const Self, ctx: *ExecContext) !?Self // refcounted view
pub fn zeroGrad(self: *const Self) void
pub fn detach(self: *const Self, ctx: *ExecContext) !Self    // no-grad view; cuts the graph
```

Both backward entries fail with `error.NoGradientGraph` on a tensor that has
no `grad_state` at all — you asked a constant to explain itself.

## 7.6 Running backward: seeds, fan-out, and one pass per graph

### Seeding

The reverse pass starts from the gradient of the loss with respect to
*itself*, which is 1. Fucina's rules (`docs/REFERENCE.md:3743-3756`):

- a **scalar** output (one element total) with no gradient present receives
  the implicit seed `1`;
- a **non-scalar** output with no gradient present fails with
  `error.MissingOutputGradient` — there is no canonical "the" gradient of a
  vector output, so you must supply the cotangent yourself with
  `backwardWithGrad(ctx, grad_output)` (same tags, shape-checked, read as a
  value);
- an output that **already holds** a gradient is respected as-is — the
  implicit `+1` is never added on top.

The machine-verified `test "non-scalar output needs a seed"`
(`docs/REFERENCE.md` §5.2) walks the failure and the recovery: `y = 2x` is a
two-element output, so a bare `y.backward(&ctx)` fails with
`error.MissingOutputGradient` — *before any scheduling state exists*, so the
same graph stays runnable — and
`y.backwardWithGrad(&ctx, &grad_output)` with cotangent `{1, 10}` then
delivers `x.grad == {2, 20}`: each output element's sensitivity, weighted by
your seed.

> **ML note** — Seeding with an arbitrary vector is not a workaround; it is
> the definition. `backwardWithGrad(v)` computes `vᵀ·J` — the
> vector–Jacobian product of the *whole program*; the scalar-loss case is
> just `v = [1]`, and a one-hot seed extracts one row of the Jacobian.

### Fan-out, for real this time

The toy's `prepare` reappears almost verbatim, now with an atomic counter
because contributions may arrive concurrently (`src/ag/core.zig:173-184`):

```zig
    fn prepareBackwardPass(self: *GradState) void {
        _ = self.pending_grads.fetchAdd(1, .monotonic);
        if (!self.compareState(.idle, .pending)) {
            return;
        }

        if (self.grad_fn) |function| {
            for (function.operands()) |operand| {
                if (operand) |state| state.prepareBackwardPass();
            }
        }
    }
```

One walk from the output, one increment per consumer edge, recursion into
parents only on the `idle → pending` first visit. And the drain
(`src/ag/core.zig:273-277`):

```zig
    fn finishGradContributionReady(self: *GradState) bool {
        const old = self.pending_grads.fetchSub(1, .acq_rel);
        std.debug.assert(old > 0);
        return old == 1;
    }
```

When `old == 1` the last contribution just landed and the node is ready to
fire. The `x² + 4x` test from the toy exists at tensor rank as a
machine-verified snippet (`docs/REFERENCE.md` §5.2):

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

Accumulation itself is more careful than `grad += g`: contributions add in
place under the per-state `grad_mutex`, and before mutating an accumulator
the engine checks the raw tensor's exclusive-ownership predicate,
materializing a private copy when the buffer is shared — so a VJP may hand
back cheap refcounted *views* of `gy` without risking cross-state aliasing
(`docs/REFERENCE.md:3818-3824`; pinned in `src/ag/core_tests.zig`).

And the `needs_grad` pruning promised earlier: each VJP receives one bool
per operand slot — true only where the slot has a `GradState` — so gradients
for constants (frozen weights, masks, cached KV) are *never computed*
(`docs/REFERENCE.md:3826-3829`). This is what makes LoRA-style fine-tuning
over a frozen quantized base cheap ([Chapter 15](15-training-llms-on-cpu.md)):
the mountain of frozen weights contributes zero backward work.

### One backward per graph

A completed pass leaves gradients accumulated in *every* `GradState` it
touched — interior op results included, not just leaves. Re-running backward
over the same retained graph would compound: interior states would receive
fresh contributions on top of their previous gradients, which then flow
downstream multiplied — two passes over `loss = sum(x·x)` would yield
`3·(2x)`, not `2·(2x)` (`docs/REFERENCE.md:3863-3876`).

So the engine refuses: a completed pass marks its outputs consumed
(`backward_done` in the struct above), and a repeat fails with
`error.BackwardAlreadyRun` before installing any scheduling state. Two
things to keep straight:

- **`zeroGrad` resets gradients, not the consumed graph.** The supported
  idiom is one backward per *freshly built* forward graph — forward is cheap
  to rebuild; it is eager code you already wrote.
- Gradient **accumulation across steps** is still first-class: build a fresh
  graph per micro-batch over the *same leaf variables* and call backward once
  per graph; leaf gradients sum, and `zeroGrad` resets them between optimizer
  steps. The machine-verified `test "micro-batch accumulation and zeroGrad"`
  in `docs/REFERENCE.md` §5.3 shows the idiom in eight lines: two fresh
  graphs over one leaf `w`, two backwards, `w.grad()` holds the sum.

### Failure atomicity

A design habit worth stealing: order operations so an error exit leaves zero
debris. Seeds are validated and pre-allocated *before* any pending counter is
installed, and the comment at `src/ag/core.zig:543-547` says why: "an error
exit after `prepareBackwardPass` would strand nonzero counters, and the next
backward over the same states would stop at their `.pending` check and
report success with missing gradients." That is why the unseeded
`MissingOutputGradient` failure above left a graph that could retry
successfully. When a VJP fails mid-pass, the engine deinits the gradients it
produced, restores the failing node's operand counters, drains in-flight
tasks, and returns the first error — re-runnable again. One honest caveat:
re-runnability restores *scheduling* state, not values; contributions
delivered before the failure remain accumulated, so call `zeroGrad` on the
leaves before retrying if exact values matter
(`docs/REFERENCE.md:3849-3861`).

## 7.7 Draining counters on a bounded pool

Now the second half of the Origins quote. spaGO ran backward the Go way: one
goroutine per graph node, each blocked until its contributions arrived, the
runtime scheduler absorbing the wait. Zig has no goroutines and no runtime,
so the idea was *rethought rather than translated*: readiness became data. A
counter draining to zero — the `old == 1` above — is a fact you can act on
immediately, on whatever thread observed it. No node ever waits; it simply
doesn't exist as a task until it is ready.

When several independent nodes become ready at once, the engine may fan
them out to Chapter 6's worker team (`src/ag/core.zig:421-433`):

```zig
    fn scheduleReadyBatch(self: *GradEngine, states: []const *GradState) void {
        var async_candidates: usize = 0;
        for (states) |state| {
            if (self.isAsyncCandidate(state)) async_candidates += 1;
        }

        var async_to_spawn = if (async_candidates > 1) async_candidates - 1 else 0;
        for (states) |state| {
            const spawn = async_to_spawn > 0 and self.isAsyncCandidate(state);
            if (spawn) async_to_spawn -= 1;
            self.scheduleReadyMode(state, spawn);
        }
    }
```

Note the "all but one" policy: the scheduling thread keeps one ready node
for itself instead of handing everything off and going idle. Combined with
the counter discipline — no node exists as a task until it is ready — this
is the README's "bounded pool, no blocked workers", made concrete.

Node-level spawning is deliberately conservative, and the details matter
(`docs/REFERENCE.md:3830-3847`):

- A record is a spawn candidate only if it advertises enough work through
  the vtable: an `estimated_work()` at or above
  `parallel.backward_async_work_threshold` (`256 * 1024 * 1024` work units,
  `src/parallel.zig:31`). The providers are the heavyweights: attention, the
  causal-conv1d family, gather, linear-cross-entropy, `Conv1d`/`Conv2d`,
  `Dot`, and the ternary-STE-dot records. The `prefer_async_backward` flag
  exists as a second opt-in, but no in-tree record currently sets it — a
  seam, not an active mechanism.
- The whole feature is gated at comptime by
  `exec.parallel_dot_backward_branches` (`src/exec.zig:38`: native backend
  with BLAS). On the scalar backend, or a no-BLAS native build, every node
  runs inline on the calling thread.
- `backwardGradSerial` forces the pool off for a whole pass regardless;
  kernel-level `parallelChunks` parallelism *inside* a VJP is unaffected
  (Chapter 6's layer, orthogonal to this one). Serial mode exists for
  callers whose correctness needs the pass on one thread — the in-tree case
  is activation checkpointing (§7.9).

Small ops stay inline because a pool handoff costs more than it saves; a
`Dot` backward over a large GEMM is worth shipping to another core. Either
way, correctness never depends on the choice — the per-state mutex,
copy-on-write, and the counter discipline are the same whether contributions
arrive from one thread or eight.

> **Zig note** — Look back at the orderings: `fetchAdd(1, .monotonic)` in
> the prepare walk (only the count matters, no data is published), but
> `fetchSub(1, .acq_rel)` in the drain — the thread that observes `old == 1`
> must also observe every gradient write that preceded the other threads'
> decrements. Zig makes you name the ordering at every atomic operation;
> there is no default to hide behind. The `idle/pending/ongoing` state
> machine runs on `cmpxchgStrong` over an `enum(u8)`, bridged with
> `@intFromEnum`/`@enumFromInt` (`src/ag/core.zig:374-389`).

## 7.8 Reading gradients, noGrad, and the lifetime rule

### Reading and resetting

After `backward()`, gradients sit in the leaves — two accessors, one
distinction that matters (`docs/REFERENCE.md:3880-3896`):

- `grad(ctx)` returns a **deep copy** — caller-owned, safe to keep across
  later passes. `null` for constants and for variables with no accumulated
  gradient.
- `gradView(ctx)` is the zero-copy variant, and it aliases the accumulator
  **as of that moment** — it is *not* a live window onto future passes: the
  held reference defeats the engine's copy-on-write check, so a later
  backward accumulates into a fresh private buffer and the view silently
  keeps the stale pre-pass value. `gradView` for immediate reads, `grad` for
  anything that must observe later passes.

`zeroGrad()` frees the accumulated gradient (no-op on constants); training
loops call it between optimizer steps. `detach(ctx)` returns a no-grad
constant sharing the same storage — the value flows on, the graph is cut.
And one pleasant guard: `data()` refuses mutable access on a grad-carrying
tensor with `error.MutableDataRequiresNoGrad` — mutating a recorded value
would silently invalidate the graph; `dataConst()`/`item()` are always
allowed (`docs/REFERENCE.md:3902-3905`).

### Evaluation mode: noGrad

Sometimes you have variables but want no graph — validation passes, greedy
decoding mid-training. The mechanism is a threadlocal integer
(`src/ag/control.zig:4-25`):

```zig
threadlocal var no_grad_depth: usize = 0;
// ... (a sibling threadlocal for the file's second scope elided)

pub const NoGradScope = struct {
    active: bool = true,

    pub fn close(self: *NoGradScope) void {
        if (!self.active) return;
        std.debug.assert(no_grad_depth > 0);
        no_grad_depth -= 1;
        self.active = false;
    }
};

pub fn noGrad() NoGradScope {
    no_grad_depth += 1;
    return .{};
}

pub fn isGradEnabled() bool {
    return no_grad_depth == 0;
}
```

That is the whole mechanism — nesting via a depth counter, per-thread via
`threadlocal`, idempotent `close` via the `active` flag (so an early close
composes with a `defer scope.close()`); the rest of the 75-line file is a
second scope of the same shape (a GPU quant-dot disable switch) plus the
tests. `finishOp` checks
`isGradEnabled()` in its very first line — while a `noGrad` scope is open,
`x.scale(&ctx, 2)` on a variable takes the identical forward path but the
result reports `requiresGrad() == false`: no backward node was built
(machine-verified as `test "noGrad suppresses recording"`,
`docs/REFERENCE.md` §5.4).

> **ML note** — This is `torch.no_grad()`: bit-identical values, no memory
> spent on the graph. In Fucina it is also honest about its scope: the
> counter is per-thread, so a worker thread doing evaluation never disturbs
> a training thread.

### The lifetime rule — read this twice

Here is the price of the implicit graph, stated without softening. Each op
result owns two things with *different* ownership models
(`docs/TRAINING.md` §2):

- its **value** — refcounted storage. VJP records save *views*, so releasing
  a value early never dangles data;
- its **GradState** — the graph node. It is **single-owner, not refcounted**:
  the tensor owns it, `tensor.deinit()` destroys it unconditionally, and the
  downstream records hold raw `*GradState` pointers to it.

The payoff: no atomic refcount traffic on the eager hot path, no ownership
cycles, and inference — where `grad_state == null` — pays nothing. The
price, quoting `docs/TRAINING.md` §2 exactly: "deinit an intermediate before
backward and the backward pass walks a dangling node — **undefined behavior,
not an error you can catch**." UB. The rule is therefore absolute: every
tensor on the path from the parameters to the loss stays alive until
`backward()` returns.

Living under that rule sounds like keeps and errdefer chains everywhere —
and this is where the library's headline ergonomic move comes in. From the
`README.md` front page:

```zig
// Inference: the defers free each intermediate as soon as it is consumed.
// Training: open an exec scope and the SAME forward trains as-is — the
// scope adopts every intermediate (value + autograd node), each deinit
// becomes a no-op borrow-release, and the step's whole graph stays alive
// until backward(), then is released at once when the scope closes.
const scope = ctx.openExecScope();
defer ctx.closeExecScope(scope);
const logits = try forward(ctx, &model, &x);
const loss = try logits.crossEntropy(ctx, .class, labels);
try loss.backward(ctx);
try opt.step(ctx);
```

You saw the mechanism in `finishOp`: while a scope is open, every op result
is adopted by the innermost scope and the caller receives a borrow whose
`deinit` is a safe no-op (`scope_owned`). Forward code written once with
inference idioms — defer-deinit every intermediate — *trains unchanged*
under a scope. Scopes nest with stack discipline, a failed op mid-forward
leaks nothing (the scope already owns the prefix), and tensors you create
explicitly plus gradients you fetch stay yours (`docs/TRAINING.md` §2). The
one hazard the borrow flag cannot remove: never *use* a scope-owned tensor
after its scope closes.

Two footnotes on the same theme. First, *composed* facade ops (`nllLoss`,
`select`, `stack`, `einsumMany`, and friends — full list at
`docs/REFERENCE.md:3960-3968`) build function-local graph nodes they cannot
own past their return, so when gradients are tracked they demand an active
scope and fail loudly with `error.ActiveExecScopeRequired` — the engine
converts the UB it *can* detect into an error. Second, an explicit
per-tensor owner (`Tape`) was prototyped early and removed unused: scopes
covered its niche, and nested scopes release suffixes more finely
(`docs/TRAINING.md:109-111`). The design earned its shape.

The full training-loop choreography — optimizers, `zeroGrad` placement,
clipping, schedules — is [Chapter 8](08-training.md)'s territory.

## 7.9 Trading compute for memory: activation checkpointing

The lifetime rule has a cost: a deep network's *every* intermediate stays
alive until backward — often the binding memory constraint of training, far
ahead of the parameters themselves. Activation checkpointing is the classic
answer: *forget* the intermediates and recompute them when backward actually
needs them. Fucina implements it with its own primitives, in 435 lines
(`src/ag/checkpoint.zig`):

```zig
pub fn checkpoint(ctx: *ExecContext, comptime block: anytype, inputs: anytype)
    !BlockOutput(block, @TypeOf(inputs))
pub fn checkpointWithContext(ctx: *ExecContext, comptime block: anytype,
    extra: anytype, inputs: anytype)
    !BlockOutputWithContext(block, @TypeOf(extra), @TypeOf(inputs))
```

**Forward:** run the block on grad-free constants inside an *inner exec
scope*; closing the scope frees every block intermediate immediately, and
only refcounted views of the inputs plus one deep copy of the output are
kept — "This is the entire memory win" (`src/ag/checkpoint.zig:131-135`).

**Backward:** when the incoming gradient reaches the checkpoint node, re-run
the block on the stored input views to rebuild the subgraph, install `gy` on
the recomputed output, and run a full *nested* backward over the rebuilt
subgraph — the engine is reentrant enough to be its own building block
(`src/ag/checkpoint.zig:323-343`):

```zig
            const recomputed = try callBlock(block, ctx, self.extra, &rewrapped);
            const out_state = recomputed.grad_state orelse return error.CheckpointOutputNotDifferentiable;

            // Seed the recomputed output with the incoming gradient and run
            // a full backward over the recomputed subgraph. The SERIAL
            // variant keeps every recomputed node on this thread: a nested
            // checkpoint node scheduled onto a pool thread would pass the
            // threadlocal `recompute_active` check and deadlock on the held
            // `recompute_mutex` instead of erroring.
            out_state.setGrad(try gy.cloneView());
            try core.backwardGradSerial(ctx, &.{out_state}, &.{recomputed.asRawTensor()});
```

There is the promised consumer of `backwardGradSerial` (§7.7), and there is
why the seeding rules of §7.6 respect a pre-installed gradient rather than
topping it up with `+1`. The bookkeeping result: **O(inputs + output)
retained per checkpoint instead of O(intermediates)**. The measured numbers,
as documented:

- an 8-block chain retains 8 scope entries versus 24 plain, with bitwise
  gradient parity — both asserted in `src/ag/checkpoint_tests.zig`
  (`docs/TRAINING.md:346-347`);
- at real-model scale, per-layer checkpointing of the Qwen3-0.6B LoRA
  fine-tune reproduced digit-identical losses at **~+8.5% step time**
  (`docs/TRAINING.md:369-371`) — a dated, machine-specific snapshot of the
  trade, not a law;
- the gradients are **bitwise identical** to the non-checkpointed forward
  (`docs/REFERENCE.md:4162-4164`) — recompute-in-backward is not an
  approximation.

The contract that buys the bitwise claim: the block must be **deterministic
and pure in its inputs**, because the recompute must rebuild the exact
forward values. Here a design decision from
[Chapter 5](05-the-operation-library.md) pays off spectacularly: dropout's
mask is a pure function of `(seed, element index)` and is never stored
(`docs/REFERENCE.md:4048-4051`), so dropout under a checkpoint replays
bitwise *by construction* — ambient RNG state would have meant "silently
wrong gradients", the documented failure mode of an impure block
(`docs/TRAINING.md` §12). Everything non-differentiable a block needs —
frozen quantized weights, RoPE tables, config — travels through
`checkpointWithContext`'s `extra`: stored by value, pointees valid until
backward, never receiving gradients. Nested checkpoints are rejected with
`error.NestedCheckpointRecompute` rather than deadlocking.

Usage is pleasantly boring (machine-verified as
`test "checkpointed layer backward"`, `docs/REFERENCE.md` §5.5):

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
```

then `var y = try fucina.checkpoint(&ctx, ckptLayer, .{ &x, &w })` in place
of a direct call — only the inputs and `y` are retained, and the block
re-runs during `loss.backward(&ctx)` to rebuild its subgraph. The body is
the ordinary defer-deinit forward idiom: it always runs under a scope.

## 7.10 Extending the engine: customVjp and elemental ops

Every built-in op has a hand-written VJP record, but the engine is open at
two levels.

**`customVjp`** (`src/ag/custom.zig:38`, re-exported at the root) admits a
fully custom differentiable op. You provide a `Spec` — an output type, a
forward on raw tensors, a backward implementing your VJP — and the adapter
does the plumbing: refcounted views of inputs and output, `noGrad` honored,
`needs_grad` pruning, shape checks on every gradient you return. From
`docs/REFERENCE.md` §5.6:

```zig
const RawTensor = fucina.internal.RawTensor;

const ScaledSquare = struct {
    pub const Output = fucina.Tensor(.{.d});

    pub fn forward(ctx: *fucina.ExecContext, extra: f32, inputs: []const *const RawTensor) !RawTensor {
        var sq = try ctx.mulRank(1, inputs[0], inputs[0]);
        defer sq.deinit();
        return ctx.scale(&sq, extra); // y = extra * x^2
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
            var slope = try ctx.scale(inputs[0], 2 * extra); // dy/dx = 2*extra*x
            defer slope.deinit();
            out[0] = try ctx.mulRank(1, gy, &slope); // engine consumes out[0]
        }
    }
};
```

Missing declarations are compile errors — the `Spec` contract is checked
with `@hasDecl` at comptime, so a typo'd `backward` never becomes a runtime
mystery.

**Elemental ops** are the convenience tier above it (`src/ag/elemental.zig`,
surfaced as `elementalUnary`/`elementalBinary` on the facade,
`docs/REFERENCE.md` §4.4): for a pointwise function you write scalar math
only — the adapter owns buffers, broadcasting, gradient sum-reduction, and
the worker-team chunking:

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

Note the contract: you return the *propagated* gradient `dL/dx = dy/dx · gy`,
not the bare local derivative — the same VJP framing, at scalar granularity.

> **ML note** — VJPs don't have to be true derivatives. Fucina's
> `dotTernarySte` implements a *straight-through estimator*: forward uses
> the quantized ternary weight, backward pretends the quantization wasn't
> there — "dx through the quantized weight, dW as-if-unquantized"
> (`docs/REFERENCE.md` §5.8). A deliberate lie that makes non-differentiable
> quantization trainable; [Chapter 14](14-the-low-bit-frontier.md) builds on
> it.

## 7.11 Verified, not trusted: gradcheck and the verification ladder

A wrong gradient is the worst kind of bug: nothing crashes, the loss still
goes down (mostly), and your model quietly trains worse than it should. So
gradients here are never *trusted* — they are checked against an oracle that
cannot share their bugs: central finite differences. Perturb one input
element by `±eps`, evaluate the loss twice, and the slope
`(f(x+ε) − f(x−ε)) / 2ε` approximates the true derivative with `O(ε²)` error
— no calculus, no shared code path with the VJPs, just subtraction.
`fucina.gradcheck` (`src/ag/gradcheck.zig`, re-exported at the root) runs
this for every element of every variable input against the analytical
backward:

```zig
pub fn gradcheck(ctx: *ExecContext, comptime loss_fn: anytype,
                 inputs: anytype, options: Options) !Result
```

| `Options` field   | default | meaning                                     |
|-------------------|---------|---------------------------------------------|
| `eps`             | `1e-3`  | central-difference step (`f64`)              |
| `abs_tol`         | `1e-3`  | absolute tolerance floor                     |
| `rel_tol`         | `1e-2`  | relative tolerance factor                    |
| `print_mismatch`  | `true`  | print the first failing element              |

Per element the criterion is `|g_num − g_ana| ≤ abs_tol + rel_tol·|g_ana|`
(`docs/REFERENCE.md` §5.7) — a mixed tolerance in the torch style, because a
pure relative test explodes near zero and a pure absolute test is
meaningless for large gradients. The differencing runs in `f64` to keep the
subtraction of nearly equal numbers from eating the signal. Even the
perturbation is failure-atomic: the loop body at
`src/ag/gradcheck.zig:76-82` opens with `errdefer param.* = original;`, so a
loss evaluation that fails mid-check restores the perturbed parameter — the
same zero-debris habit as §7.6's seeding.

Closing the loop from §7.10 — validate the custom op with the oracle
(machine-verified, `docs/REFERENCE.md` §5.7):

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

This is the pattern for every new VJP: `src/ag/gradcheck_tests.zig` runs the
same oracle over built-ins from conv2d to the losses and norms. Write the
rule, then make arithmetic vouch for it.

### The ladder at model scale

Element-wise gradcheck is the ground floor. For the production Qwen3 LoRA
trainer the repo documents three *independent* verification angles on top of
it — full-stack finite differences through the entire `Trainer.loss`, an
independent PyTorch golden replica, and causal checks on the real quantized
model — and one leg of the third angle is *red* and documented as such: a
quantization-sensitive first-order Taylor probe with an open, not yet
root-caused discrepancy (`docs/TRAINING.md` §9, "Gradient verification").
[Chapter 15](15-training-llms-on-cpu.md) walks that ladder in detail,
numbers included. A verification section that records its own open failure
is worth more than one that only reports green — you know the other entries
mean something.

## What you now know

- Reverse-mode AD gets every gradient from one backward sweep; each op
  contributes only a local VJP rule ("given `dL/dy`, produce `dL/dx`"), and
  composing the rules backward *is* the chain rule.
- Fucina's graph is **implicit in the values** (the spaGO-inherited idea):
  each result's `GradState` points at a VJP record, which points at the
  operand states. No tape, no graph object, no central engine — and a guard
  test keeps the deleted heavyweight design from coming back.
- Naive recursive backward double-counts on any DAG; the fix is a **pending
  counter** — count consumer edges in a prepare walk, fire a node only when
  its counter drains. You built the scheme in ~60 lines of scalar Zig; the
  real engine is recognizably the same code with atomics.
- Everything funnels through `finishOp`: constants stay graph-free,
  variables are leaves, forward always takes the identical kernel path — so
  the same forward code trains when you open an exec scope around it.
- Scalar outputs seed implicitly with 1; non-scalar outputs need
  `backwardWithGrad` (a genuine cotangent). One backward per freshly built
  graph: a repeat is `error.BackwardAlreadyRun`, and `zeroGrad` resets
  gradients, *not* the consumed graph.
- `GradState` is single-owner: deinit an intermediate before backward with
  no scope open and you get undefined behavior, not an error. Exec scopes
  are the ergonomic answer; composed ops demand one loudly.
- Backward drains atomic counters onto a bounded worker pool — "all but one"
  spawning, opt-in by estimated work, no blocked workers;
  `backwardGradSerial` exists for passes that must stay on one thread.
- Activation checkpointing trades ~+8.5% step time (measured, Qwen3-0.6B
  LoRA) for O(inputs + output) retained memory, with *bitwise identical*
  gradients — enabled by deterministic, seed-derived dropout.
- Gradients are verified, not trusted: `gradcheck` finite differences for
  every op, then full-stack FD, PyTorch goldens, and real-model causal
  checks — including one honestly documented open discrepancy in the
  quantization-sensitive Taylor probe.

## Explore the source

- `src/ag/core.zig` — the whole engine in 733 lines: `GradState`,
  `BackwardFunction`, the pending-counter walk, the pool scheduler. Read it
  next to your toy.
- `src/ag/backward.zig` — the VJP inventory. Start at `ReluBackward`
  (line 472), then find your favorite op's rule.
- `src/ag/tensor.zig` — `finishOp` (line 5673) and the gradient accessors
  (lines 891–974): where the facade meets the engine.
- `src/ag/control.zig` — `noGrad` (plus a sibling GPU quant-dot scope) in
  75 lines; the cheapest file in the chapter and a model of scope-handle
  design.
- `src/ag/checkpoint.zig` — recompute-in-backward built from the library's
  own primitives; the engine using itself.
- `src/ag/gradcheck.zig` — the finite-difference oracle.
- `docs/REFERENCE.md` §5 and `docs/TRAINING.md` §2/§9 — the authoritative
  reference (every test snippet machine-verified by `zig build
  snippet-check`), the lifetime rules, and the verification ladder.

## Exercises

1. **Warm-up.** Extend the course-code scalar autograd with a `tanh` rule
   (`d/dx tanh(x) = 1 − tanh²(x)`; the forward *output* is exactly what
   backward needs — save it, as many real VJPs do). Verify `d/dx tanh(x·y)`
   at a couple of points by hand.
2. **Your own oracle.** Add a finite-difference checker to the toy: perturb
   `data` by `±1e-3`, re-run the forward, compare the slope against `grad`
   with the mixed tolerance from §7.11. You now have a miniature `gradcheck`
   — use it on exercise 1.
3. **Constants.** Give the toy a `requires_grad: bool` and mimic Fucina's
   `needs_grad` pruning: constants neither accumulate `grad` nor propagate.
   Instrument a counter of rule firings and check that a "frozen" operand
   adds zero backward work.
4. **One backward per graph.** Run the toy's backward twice over a retained
   `loss = x·x` graph and show the leaf gradient becomes `3·(2x)`, exactly
   as `docs/REFERENCE.md:3863-3876` predicts; then add a `backward_done`
   guard, and explain why a leaf-only `zeroGrad` would not have been enough
   (interior nodes hold gradients too).
5. **Real engine.** Write a `customVjp` spec for softplus
   (`y = log(1 + eˣ)`, `dy/dx = sigmoid(x)`), validate it with
   `fucina.gradcheck`, then reimplement it as an `elementalUnary` op and
   compare the amount of code. Read `src/ag/elemental.zig` to see what the
   adapter did for you.

---

[Previous: Going fast on CPUs](06-going-fast-on-cpus.md) · [Next: Training: making the machine learn](08-training.md)
