# Chapter 01 — Just enough Zig

*Part I — Foundations*

This is not a Zig course. It is exactly the Zig you need to read — and, over
the coming chapters, rebuild — a tensor library. Every feature earns its
place the way it earned its place in Fucina: because a tensor runtime needed
it. Allocators, because every tensor buffer must be accounted for. Error
unions, because shape mismatches are values, not exceptions. Functions that
return types, because `Tensor(.{ .batch, .in })` being a *different type*
from `Tensor(.{ .in, .batch })` is the library's signature idea.

The strategy: start from one real, twenty-seven-line program — the library's
canonical smoke test — and unfold every language feature in it into its own
section. By the end of the chapter you can read that program the way its
author does. By the end of the course you could have written it.

If you already know Zig, skim the openers and read the repo excerpts — they
show how the language is *used* here, which is not always how tutorials use
it. If you come from Python and PyTorch, read everything; the `> **Zig
note**` asides are for you.

## 1.1 Setting up the forge

Fucina is pinned to one exact compiler version. Not "0.16 or newer" — exactly
0.16.0:

> Fucina is pinned to **Zig 0.16.0** — `zig version` must print `0.16.0`;
> other versions do not build.
> — `docs/REFERENCE.md` §2.1

Download it from [ziglang.org/download](https://ziglang.org/download/) (the
toolchain is a single archive — unpack it and put `zig` on your `PATH`; there
is nothing else to install). Then:

```sh
zig version        # must print 0.16.0
git clone https://github.com/matteo-grella/fucina
cd fucina
zig build test     # all test roots; no model files needed
```

`zig build test` compiles and runs the unit tests of nine separate test roots
— the tensor core, the LLM stack, and seven application examples — and all of
them pass with no model assets on disk (`docs/REFERENCE.md` §2.7). If that
command is green, your forge is lit.

There is deliberately no package manifest: no `build.zig.zon`, no lock file.
Every module and option is wired directly in `build.zig` (`AGENTS.md`,
Toolchain). You clone, you build; §1.16 explains how.

For a first taste of the whole machine, run the training demo:

```sh
zig build spirals -Doptimize=ReleaseFast
```

It trains a small neural network to separate two interleaved spirals — once
per optimizer — printing a loss and accuracy line for each, plus a
checkpoint-resume check that must come out *bit-exact*
(`examples/spirals.zig`). [Chapter 08](08-training.md) walks through every
line; for now it is proof that a complete train-checkpoint-infer loop lives
in one ordinary Zig file.

One warning before you read any Zig on the internet: **version 0.16 broke
things**. Most online tutorials describe an older Zig whose code no longer
compiles. The repo keeps a list of the traps (`AGENTS.md`, "Zig 0.16
notes"); §1.17 summarizes it. When in doubt, imitate this repo's code, not a
blog post.

## 1.2 The whole library in twenty-seven lines

Here is the program this chapter unfolds. It is `docs/REFERENCE.md` §1.4,
"A first program", quoted verbatim — and it is *machine-verified*: a CI step
(`zig build snippet-check`) extracts every runnable snippet in that document
and runs it against the real library, so this code cannot silently rot:

```zig
const std = @import("std");
const fucina = @import("fucina");

test "first program" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // x: [batch=1, in=2], w: [in=2, out=1]
    var x = try fucina.Tensor(.{ .batch, .in }).variable(&ctx, try ctx.fromSlice(&.{ 1, 2 }, &.{ 2, 3 }));
    defer x.deinit();
    var w = try fucina.Tensor(.{ .in, .out }).variable(&ctx, try ctx.fromSlice(&.{ 2, 1 }, &.{ 4, 5 }));
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in); // contract .in => [batch, out]
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?; // dloss/dx = w^T = [4, 5]
    defer gx.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 23.0), try loss.item(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), (try gx.dataConst())[0], 1e-6);
}
```

In ML terms: build a 1×2 input `x` and a 2×1 weight matrix `w`, multiply
them (contracting the axis named `.in`), sum the result into a scalar loss
(2·4 + 3·5 = 23), and ask autograd for the gradient of that loss with
respect to `x` — which the comment works out by hand: `w` transposed,
`[4, 5]`. Don't worry if the ML half is fog;
[Chapter 02](02-just-enough-ml.md) builds it from nothing.

What it does, in Zig terms, is a checklist of this chapter:

| In the snippet | The feature | Section |
| --- | --- | --- |
| `alloc`, `ctx.init(alloc)` | allocators — memory is a parameter | 1.3 |
| `try`, the `!` hiding in every op | error unions — errors are values | 1.4 |
| `defer ctx.deinit()`, `defer x.deinit()` | deterministic cleanup | 1.5 |
| `&.{ 2, 3 }`, `dataConst()` returning `[]const f32` | arrays vs slices | 1.6 |
| `ctx.fromSlice(...)`, methods on structs | structs and methods | 1.7 |
| `(try x.grad(&ctx)).?` | optionals — "maybe" in the type | 1.8 |
| `.in`, `.batch` — bare dot-names | enum literals, enums, `switch` | 1.9–1.10 |
| `fucina.Tensor(.{ .batch, .in })` | a *function call that returns a type* | 1.11 |
| `test "first program" { ... }` | tests live in the language | 1.15 |

Notice what is *absent*: no graph builder, no `device=`, no framework
ceremony. Fucina is eager — "What you write is what runs, in the order you
wrote it" (`docs/REFERENCE.md` §1.2) — so the program reads top to bottom
like the arithmetic it performs.

> **ML note** — `x.dot(&ctx, &w, .in)` names the axis it contracts instead of
> using a positional `dim=` argument. That name, `.in`, is part of the
> *type* of `x`. Contract an axis the tensor does not have and the program
> does not crash — it does not compile. This is the library's crown jewel and
> [Chapter 04](04-axes-with-names.md) is devoted to it.

## 1.3 Allocators: memory is a parameter

Zig has no garbage collector and no hidden `malloc`. Any function that
allocates memory says so in its signature by taking an
`std.mem.Allocator` — and the *caller* decides which allocator that is: a
general-purpose heap, a debug allocator that tracks leaks, a fixed buffer, an
arena. Memory policy is dependency-injected.

This is the first thing the first program does:

```zig
const alloc = std.testing.allocator;
var ctx: fucina.ExecContext = undefined;
ctx.init(alloc);
```

`std.testing.allocator` fails the test if a single byte is still allocated
when the test ends. Fucina's unit tests run under it by convention, which
means **leak detection is not a tool you run — it is what `zig build test`
already does**. The context threads that one allocator into the whole
runtime; the signature is `pub fn init(self: *ExecContext, allocator:
Allocator) void` (`src/exec.zig:126`).

Outside tests, the demo programs use the debug allocator with an explicit
leak check — from `examples/spirals.zig:327-330`:

```zig
pub fn main(init: std.process.Init) !void {
    var gpa = std.heap.DebugAllocator(.{}){};
    defer if (gpa.deinit() == .leak) @panic("leak");
    const allocator = gpa.allocator();
```

A training demo that allocates tensors for two thousand steps, five
optimizers, and a checkpoint round-trip ends by *panicking if anything
leaked*. That is the standard the whole library holds itself to.

Here is the pattern in miniature — course code, compile-checked with
`zig test` (as is every fresh snippet in this chapter):

```zig
const std = @import("std");

fn sumOfSquares(allocator: std.mem.Allocator, n: usize) !f32 {
    const buf = try allocator.alloc(f32, n); // may fail: OutOfMemory
    defer allocator.free(buf); // paired at the declaration site

    for (buf, 0..) |*v, i| v.* = @floatFromInt(i);
    var total: f32 = 0;
    for (buf) |v| total += v * v;
    return total;
}

test "allocation is explicit and leak-checked" {
    // std.testing.allocator fails the test if one byte leaks.
    const total = try sumOfSquares(std.testing.allocator, 4);
    try std.testing.expectEqual(@as(f32, 14.0), total); // 0+1+4+9
}
```

Why a tensor library needs this: every tensor buffer, autograd node, and
thread-pool structure is accounted for, and the user — not the runtime —
chooses the memory policy. When [Chapter 03](03-tensors-from-scratch.md)
builds refcounted storage and its buffer pool, and when
[Chapter 10](10-the-guitar-amp.md) demands *zero* allocation inside a
real-time audio callback, this explicitness is what makes both possible.

> **ML note** — In PyTorch you never see an allocator; the framework hides
> memory. Convenient until it isn't: fragmentation, unpredictable latency,
> out-of-memory errors you cannot reason about. Here the memory story of a
> training step is readable in the code — and testable.

## 1.4 Errors are values: error unions and `try`

Zig has no exceptions. A function that can fail returns an **error union**,
written `!T` — "either a `T` or an error". Errors are ordinary values drawn
from named error sets. Fucina's raw tensor layer defines a small, closed one
(`src/tensor.zig:11-19`):

```zig
pub const TensorError = error{
    ShapeMismatch,
    InvalidShape,
    InvalidDataLength,
    IndexOutOfBounds,
    UnsupportedView,
    EmptySelection,
    DivisionByZero,
};
```

That is the complete list of ways a raw tensor operation can fail. Not an
exception hierarchy — seven names.

At each call site that can fail, you must do something visible:

- `try f()` — unwrap the value, or return the error to *your* caller;
- `f() catch fallback` — handle it locally;
- `f() catch |err| switch (err) { ... }` — handle each case.

That is why the first program says `try` twelve times: every fallible step is
marked, and the failure path of the whole program is legible at a glance.
Course code (tests assert error values as easily as successes):

```zig
const std = @import("std");

const ShapeError = error{ ShapeMismatch, InvalidShape };

fn elementCount(dims: []const usize) ShapeError!usize {
    if (dims.len == 0) return ShapeError.InvalidShape;
    var n: usize = 1;
    for (dims) |d| {
        if (d == 0) return ShapeError.InvalidShape;
        n *= d;
    }
    return n;
}

test "errors are values; try propagates them" {
    try std.testing.expectEqual(@as(usize, 6), try elementCount(&.{ 2, 3 }));
    try std.testing.expectError(ShapeError.InvalidShape, elementCount(&.{ 2, 0 }));
}
```

The real thing — `Shape.init` in `src/tensor.zig:25-34`, quoted in §1.7 — is
this function grown up. Why a tensor library needs this: shape errors are the
most common failure in numeric code, and here they are typed, exhaustive, and
impossible to ignore silently — there is no unchecked-exception escape hatch
for a mismatched matmul to fly through.

> **Zig note** — A bare `!T` return type means "infer my error set from
> whatever I `try`". Writing `TensorError!usize` instead *closes* the set:
> callers can `switch` on it exhaustively and the compiler knows every case.

## 1.5 `defer` and `errdefer`: cleanup you can read

`defer expr;` schedules `expr` to run when the enclosing scope exits — by any
path: normal return, error return, `break`. Multiple `defer`s run in reverse
order, like a stack. One keyword replaces destructors, `try/finally`, and
RAII, with a crucial readability gain: **the release is written next to the
acquire**. You saw the library's dominant idiom throughout the first
program:

```zig
var y = try x.dot(&ctx, &w, .in);
defer y.deinit();
```

Acquire a tensor, immediately pledge its release. `docs/MEMORY-MODEL.md`
(line 28) makes the stakes plain: "That `deinit` is not a naive free — it is
the *driver* of buffer recycling." Freed tensor buffers go back into a pool
and get reused within the same forward pass; the `defer` discipline is not
just hygiene, it is the memory system's engine. ([Chapter 03](03-tensors-from-scratch.md)
covers the pool; the arena-allocator alternative was considered and rejected,
and that document records why.)

`errdefer` is the same, but it fires **only when the scope exits with an
error**. It exists for partial construction: you have built three of six
things, the fourth fails — the first three must be freed, but only on that
failure path (on success, the caller takes ownership). Fucina's model
constructor is a textbook ladder (`examples/spirals.zig:54-66`, abridged):

```zig
var w1 = try Tensor(.{ .h1, .in }).variableFromSlice(ctx, .{ hidden, 2 }, &w1_buf);
errdefer w1.deinit();
var b1 = try Tensor(.{.h1}).variableFromSlice(ctx, .{hidden}, &b1_buf);
errdefer b1.deinit();
var w2 = try Tensor(.{ .h2, .h1 }).variableFromSlice(ctx, .{ hidden, hidden }, &w2_buf);
errdefer w2.deinit();
// ... b2, w3, b3 in the same pattern ...
return .{ .w1 = w1, .b1 = b1, .w2 = w2, .b2 = b2, .w3 = w3, .b3 = b3 };
```

If constructing `b3` fails, `w3`, `b2`, `w2`, `b1`, `w1` free themselves, in
reverse order, and the error propagates — no goto-cleanup, no leaked
half-built model. Course code proving the mechanism, with the leak-checking
allocator as the witness:

```zig
const std = @import("std");

fn makePair(allocator: std.mem.Allocator, fail_second: bool) ![2][]f32 {
    const first = try allocator.alloc(f32, 8);
    errdefer allocator.free(first); // runs ONLY if we exit with an error below

    if (fail_second) return error.OutOfMemory; // simulate the second alloc failing
    const second = try allocator.alloc(f32, 8);
    return .{ first, second };
}

test "errdefer frees partial state only on the error path" {
    // Error path: `first` must not leak — testing.allocator is the witness.
    try std.testing.expectError(error.OutOfMemory, makePair(std.testing.allocator, true));
    // Success path: errdefer did NOT run; the caller owns both.
    const pair = try makePair(std.testing.allocator, false);
    defer std.testing.allocator.free(pair[0]);
    defer std.testing.allocator.free(pair[1]);
}
```

One house convention to notice now: every `deinit` ends with
`self.* = undefined` (`src/exec.zig:132-136`; `AGENTS.md` house rules mandate
it). In Debug builds Zig fills `undefined` memory with a recognizable byte
pattern, so *using* a deinitialized struct crashes loudly instead of
corrupting quietly. A tripwire, not carelessness.

Why a tensor library needs this: a forward pass creates dozens of
intermediates per step, and `defer`/`errdefer` make each one's lifetime a
local, checkable fact. Later, *exec scopes* absorb even that
([Chapter 07](07-autograd.md)): training code needs no `defer`s at all for
intermediates.

## 1.6 Arrays, slices, and pointers: who owns the bytes

Zig separates "how many" from "whose":

| Type | Meaning |
| --- | --- |
| `[4]f32` | an **array**: a value, length fixed at compile time, copied on assignment |
| `[]f32` | a **slice**: pointer + runtime length, *mutable view* of someone else's memory |
| `[]const f32` | a read-only slice — a *borrow* you may look at but not touch |
| `*T` / `*const T` | a pointer to exactly one `T` |

Constness is enforced by the compiler, which turns an ownership doctrine into
type-checked fact. The repo states the doctrine outright: "Storage is
refcounted and owned; `[]T` slices/tensor views *borrow*"
(`AGENTS.md`, house rules). And the lowest-level compute kernels *are* their
signature — from `src/backend/vector/primitives.zig:52`:

```zig
pub inline fn vecAdd(z: []f32, x: []const f32, y: []const f32) void {
```

Mutable output, immutable inputs. A kernel cannot scribble on its inputs
without the compiler objecting.

The array/slice split also encodes *when the length is known*. Look at a
tensor constructor (`src/ag/tensor.zig:244`):

```zig
pub fn fromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self
```

The shape is `[tensor_rank]usize` — an **array**, because rank is a
compile-time fact of the tensor's type. The data is `[]const f32` — a
**slice**, because how many numbers you pass is a runtime fact. One
signature, and the design's compile-time/runtime boundary is visible in it.
The same trick shapes the raw tensor itself: `Shape` stores
`dims: [max_rank]usize` plus a runtime `len` (`src/tensor.zig:21-23`,
`max_rank = 8`) — no allocation, because rank is bounded and small.

Course code:

```zig
const std = @import("std");

// The backend-kernel signature shape: mutable output, immutable inputs.
fn axpy(z: []f32, a: f32, x: []const f32, y: []const f32) void {
    for (z, x, y) |*zi, xi, yi| zi.* = a * xi + yi;
}

test "arrays are values, slices are views" {
    const x = [_]f32{ 1, 2, 3, 4 }; // [4]f32 — the length is in the type
    const y = [_]f32{ 10, 20, 30, 40 };
    var z: [4]f32 = undefined;

    axpy(&z, 2.0, &x, &y); // &array coerces to a slice
    try std.testing.expectEqual(@as(f32, 12.0), z[0]);

    const tail = z[2..]; // a view into z: no copy, length known
    try std.testing.expectEqual(@as(usize, 2), tail.len);
    try std.testing.expectEqual(@as(f32, 36.0), tail[0]);
}
```

> **Zig note** — `for (z, x, y) |*zi, xi, yi|` iterates several slices in
> lockstep (equal lengths — Debug builds check); `|*zi|` captures by pointer
> so the loop can write through it. Kernels all over `src/backend/` have
> this shape.

Why a tensor library needs this: a tensor *view* — a transpose, a slice of a
batch — is exactly "a borrow of someone else's buffer with its own shape".
The language's borrow-vs-own distinction is the design's
([Chapter 03](03-tensors-from-scratch.md) builds views for real).

## 1.7 Structs, methods, and files as namespaces

Zig has no classes and no inheritance. A `struct` is a type holding fields
plus any declarations you put inside it — constants, functions, other types.
"Methods" are functions whose first parameter is the struct (by value,
`*Self`, or `*const Self`); `x.foo(y)` is sugar for `T.foo(x, y)`.

The real `Shape` shows the whole vocabulary in a handful of lines
(`src/tensor.zig:21-53`, abridged — it also has an `initStrides` variant and
`slice()`/`at()` accessors):

```zig
pub const Shape = struct {
    len: u8,
    dims: [max_rank]usize = undefined,

    pub fn init(values: []const usize) !Shape {
        if (values.len == 0 or values.len > max_rank) return TensorError.InvalidShape;

        var out = Shape{ .len = @intCast(values.len) };
        for (values, 0..) |value, i| {
            if (value == 0) return TensorError.InvalidShape;
            out.dims[i] = value;
        }
        return out;
    }
};
```

Fields with default values (`dims` defaults to `undefined` — deliberately
uninitialized until `init` fills it), a fallible constructor. No base class,
no framework.

Two more struct facts carry Fucina's architecture:

**A model is just a struct of tensors.** The spirals demo's entire network is
(`examples/spirals.zig:29-35`):

```zig
const Model = struct {
    w1: Tensor(.{ .h1, .in }),
    b1: Tensor(.{.h1}),
    w2: Tensor(.{ .h2, .h1 }),
    b2: Tensor(.{.h2}),
    w3: Tensor(.{ .class, .h2 }),
    b3: Tensor(.{.class}),
    // ... initRandom / initConstZero / deinit ...
};
```

No `nn.Module`, no registration decorators. (When training needs to
enumerate parameters, a comptime-reflective registry walks these fields by
name — `src/param_registry.zig`, met properly in
[Chapter 08](08-training.md).)

**Files are structs.** `@import("dtype.zig")` returns a value — the file
itself, as a namespace — and you bind it with `const`. Fucina's entire module
system is this one feature applied at scale; the public API is literally a
file of re-exports (`src/fucina.zig:15-27`, excerpt):

```zig
pub const gguf = @import("gguf.zig");
pub const optim = @import("optim.zig");
...
pub const Tensor = ag.Tensor;
```

`fucina.optim.AdamW`, `fucina.gguf`, `fucina.Tensor` — the dotted paths you
will use all course are struct field access on imported files.

## 1.8 Optionals: "maybe" in the type

`?T` is "a `T` or `null`" — and the compiler will not let you touch the `T`
without deciding what happens when it is null. Three unwrapping tools:

- `x orelse fallback` — use a default;
- `if (x) |v| { ... }` — run a branch with the unwrapped value;
- `x.?` — assert non-null (Debug builds crash if you are wrong).

The first program used the third form: `(try x.grad(&ctx)).?` — "I *know*
backward has run, give me the gradient." The reason `grad` returns an
optional is the most instructive field in the library
(`src/ag/tensor.zig:204-205`):

```zig
value: RawTensor,
grad_state: ?*GradState = null,
```

A tensor either carries gradient bookkeeping or it does not — and the type
says so. "Trainable parameter" versus "frozen constant" is not a boolean
convention or a runtime mode; it is `?*GradState`, null or not. Inference is
literally the model without grad state ([Chapter 07](07-autograd.md) builds
`GradState` itself).

Optionals also give APIs honest defaults. The SIMD width query returns an
optional — some targets have no vectors — and the kernel layer picks a floor
(`src/backend/vector/common.zig:24`):

```zig
pub const vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;
```

And "unspecified end of a slice range" is `end: ?isize = null`
(`src/ag/tensor.zig:161-165`) — not a magic sentinel like `-1`, an actual
absence. Course code:

```zig
const std = @import("std");

const Node = struct {
    grad: ?f32 = null, // "has a gradient or not" is a type fact
};

test "optionals must be unwrapped" {
    var n = Node{};
    try std.testing.expectEqual(@as(f32, 0.0), n.grad orelse 0.0);
    n.grad = 2.5;
    if (n.grad) |g| try std.testing.expectEqual(@as(f32, 2.5), g);
    try std.testing.expectEqual(@as(f32, 2.5), n.grad.?); // assert non-null
}
```

> **Zig note** — `?*GradState` is an *optional pointer*, which compiles to a
> plain pointer with null as the zero address — no space overhead. The safety
> is entirely in the type checker.

## 1.9 Enums and exhaustive `switch`

A Zig enum is a closed set of names. Fucina's dtype universe is one enum with
37 members (`src/dtype.zig:3-41`) — `bool`, the integers, `f16`/`bf16`/`f32`/
`f64`, and the whole zoo of block-quantized formats (`q4_k`, `q8_0`, ...,
`tq2_0`) that [Chapter 11](11-model-files-and-quantization.md) decodes.

The power move is `switch`: **a `switch` over an enum must handle every
member, or it does not compile**. Add a 38th dtype and every switch you
forgot to update becomes a compile error pointing at itself. The repo
weaponizes this deliberately: "prefer exhaustive `switch` over dtype/backend
so adding a variant forces edits everywhere" (`AGENTS.md`, house rules).

The most consequential switch in the library selects the compute backend —
at compile time, from a build option (`src/backend.zig:104-112`):

```zig
pub const Kind = enum {
    scalar,
    native,
};

pub const active_kind: Kind = switch (build_options.backend_kind) {
    .scalar, .cpu => .scalar,
    .native => .native,
};
```

Two things to read off this. First: Fucina has exactly **two** CPU backends —
`scalar`, the slow, obvious reference implementation that serves as the
correctness oracle, and `native`, the fast SIMD one that must always agree
with it (`cpu` is a deprecated alias for `scalar`, mapped away right here).
Second: this switch runs *in the compiler* — a few lines below, the same
pattern selects which implementation file even gets analyzed, so the backend
you did not choose is not in your binary at all. That is §1.10's subject.

Course code:

```zig
const DType = enum { f32, f16, q8_0 };

fn bytesPerElement(dt: DType) f32 {
    return switch (dt) { // exhaustive: adding a member breaks this switch
        .f32 => 4.0,
        .f16 => 2.0,
        .q8_0 => 34.0 / 32.0, // 32 weights stored in 34 bytes
    };
}
```

(That `34.0 / 32.0` is real: a Q8_0 block stores 32 weights in 34 bytes —
`src/dtype.zig:74-77`, properly decoded in Chapter 11.)

> **ML note** — dtype dispatch is the plumbing of every framework, usually
> as string comparisons or virtual calls. Here it is a closed enum the
> compiler audits — you *cannot* ship a kernel that forgot a dtype.

## 1.10 `comptime` I: code that runs in the compiler

Here is the idea that separates Zig from everything you have used: **any
expression can be evaluated at compile time, and the language is the same
language there**. No template dialect, no macros, no `constexpr` subset —
loops, functions, structs, string comparison, all executed by the compiler
when you ask (`comptime`) or when context demands it.

You have been looking at comptime values all along: `vector_len` is a
`comptime_int`; build options are comptime constants (§1.16). The axis tags —
`.batch`, `.in`, `.out` — are the flagship: the entire tag algebra is
comptime data, and the docs state the consequence plainly:

> The tag algebra is comptime-only data — it compiles down to stride
> manipulation on the raw tensor with zero runtime tagging cost.
> — `docs/REFERENCE.md` §1.2

Zero cost is meant literally: at runtime there are no tag objects, no name
lookups — nothing. The names exist only in the compiler, where they steer
which strides the generated code manipulates.

So what *is* `.batch`? A bare `.name` in Zig is an **enum literal** — a
value whose type is inferred from context. And enum literals have a type you
can capture. The first "Zig can do THIS" moment — `src/tags.zig:4`, the line
the whole named-axis system stands on:

```zig
pub const Tag = @TypeOf(.tag);
```

Read it twice. `.tag` is an enum literal; `@TypeOf(.tag)` is *the type of
enum literals themselves* — a comptime-only type whose values are names. One
line, and now functions can take axis names as parameters: `pub fn
tagIndex(comptime tags: anytype, comptime tag: anytype) ?usize`
(`src/tags.zig:178`), `x.dot(&ctx, &w, .in)`.

How do you compare two names at compile time? Character by character, in the
compiler — `src/tags.zig:190-201`:

```zig
pub fn tagEqual(comptime a: anytype, comptime b: anytype) bool {
    return comptime blk: {
        const a_name = @tagName(a);
        const b_name = @tagName(b);
        if (a_name.len != b_name.len) break :blk false;
        var i: usize = 0;
        while (i < a_name.len) : (i += 1) {
            if (a_name[i] != b_name[i]) break :blk false;
        }
        break :blk true;
    };
}
```

An ordinary string comparison — `@tagName` yields the name — forced to run
during compilation by the `comptime` keyword (the `blk:` label makes a block
an expression: `break :blk value` returns from it). Every `x.dot(&ctx, &w,
.in)` makes the compiler run loops like this to find `.in` among `x`'s tags —
and by runtime, all of it has evaporated into a fixed stride pattern.

(Writing your own miniature — `const Tag = @TypeOf(.tag);` plus a one-line
`std.mem.eql`-based `tagEqual` — is exercise territory.)

> **Zig note** — comptime execution is not free; it is honest. Heavy
> comptime computation must raise its own budget with `@setEvalBranchQuota`
> (`src/tags.zig` does, in its `unionTags` machinery) — the compiler makes
> you acknowledge real work.

## 1.11 `comptime` II: functions that return types

Types are comptime values. Therefore a function can *return* one. That
sentence is the entirety of Zig's generics — there is no separate template
syntax, no `<T>`; a generic type is an ordinary function, evaluated at
compile time, that happens to return `type`.

Course code first:

```zig
const std = @import("std");

// Course code: a function that returns a type — the whole of Zig generics.
fn FixedVec(comptime n: usize, comptime T: type) type {
    return struct {
        data: [n]T,

        pub const len = n; // a comptime constant on the type itself

        fn sum(self: *const @This()) T {
            var total: T = 0;
            for (self.data) |v| total += v;
            return total;
        }
    };
}

test "types are comptime values" {
    const V3 = FixedVec(3, f32);
    const V4 = FixedVec(4, f32);
    try std.testing.expect(V3 != V4); // distinct types

    const v = V3{ .data = .{ 1, 2, 3 } };
    try std.testing.expectEqual(@as(f32, 6.0), v.sum());
    try std.testing.expectEqual(@as(usize, 3), V3.len);
}
```

`FixedVec(3, f32)` and `FixedVec(4, f32)` are different types; mixing them up
is a compile error. Now scale the idea: what if the parameters were not a
length but a tuple of *axis names*?

That is precisely the library's public API. `fucina.Tensor` is a function —
`src/ag/tensor.zig:185-190`:

```zig
pub fn Tensor(comptime tags_spec: anytype) type {
    const tensor_dtype = dtypeFromSpec(tags_spec);
    if (comptime tensor_dtype == .f32) return FloatTensor(tags_spec);
    if (comptime dtype_mod.isBlockQuantized(tensor_dtype)) return QuantizedConstantTensor(tags_spec, tensor_dtype);
    return TypedConstantTensor(tags_spec, tensor_dtype);
}
```

Six lines that carry the whole design. `Tensor(.{ .batch, .in })` calls this
function *during compilation*; it returns a freshly built struct type whose
comptime declarations record the tags (`axis_tags`, `tag_count`,
`tensor_rank` — `src/ag/tensor.zig:199-202`). `Tensor(.{ .batch, .in })` and
`Tensor(.{ .in, .out })` are as distinct as `V3` and `V4` above — which is
why contracting a tag a tensor does not have is a *compile* error, not a
runtime shape crash.

Notice the dispatch, too: the same function implements the library's sealed
dtype policy. An `.f32` spec gets the differentiable tensor family;
block-quantized dtypes get a constant inference family; everything else, a
typed-constant family. What each dtype *can do* is decided by which struct
type you receive — "enforced by the type system, not runtime checks"
(`docs/REFERENCE.md` §1.2).

Two honest caveats, so the magic stays measured. First, only tag-*level*
errors are compile-time (wrong axis name, duplicate tags, rank overflow);
axis *sizes* stay runtime values, and size mismatches surface as
`error.ShapeMismatch`. The type system carries names, not extents
([Chapter 04](04-axes-with-names.md) teaches both halves). Second, each
distinct instantiation stamps out real code — monomorphization, the same
mechanism you will meet as a *performance* tool in
[Chapter 06](06-going-fast-on-cpus.md).

## 1.12 `inline for`: loops the compiler unrolls

If tags live at comptime, how do you loop over them? `inline for` unrolls a
loop at compile time — each iteration is stamped out separately, so each may
work with *different types or comptime values*, which an ordinary runtime
loop cannot. Every tag-set function in `src/tags.zig` is built from it; the
simplest (`src/tags.zig:9-15`):

```zig
pub fn tagsEqual(comptime a: anytype, comptime b: anytype) bool {
    if (a.len != b.len) return false;
    inline for (a, 0..) |tag, i| {
        if (comptime !tagEqual(tag, b[i])) return false;
    }
    return true;
}
```

There is even a comptime bubble sort in there — `reduceAxesDescending`
(`src/tags.zig:17-35`) sorts axis indices with nested `inline while` loops,
entirely during compilation. The same tool serves runtime paths whose *trip
count* is comptime-known: `isContiguous` unrolls over the (comptime) rank so
stride checking has no loop overhead (`src/tensor.zig:72-80`). Course code:

```zig
const std = @import("std");

test "inline for unrolls over comptime-known tuples" {
    const tags = .{ .batch, .in, .out };
    const names = comptime blk: {
        var s: []const u8 = "";
        // Inside a comptime block a plain `for` already unrolls
        // (0.16 rejects a redundant `inline` here).
        for (tags) |t| s = s ++ @tagName(t) ++ " ";
        break :blk s;
    };
    try std.testing.expectEqualStrings("batch in out ", names);
}
```

`inline for` pairs naturally with **comptime reflection**: `@typeInfo`,
`@TypeOf`, `@hasDecl`, `@hasField` let generic code inspect the types it was
handed and branch on the answer — at compile time, so untaken branches (which
might not even type-check for this instantiation) simply vanish. Fucina uses
this to make one registration function serve four optimizer types
(`if (comptime @hasDecl(@TypeOf(opt.*), "addFallbackParam"))`,
`examples/spirals.zig:119`) and to let checkpoints name tensors by walking a
model struct's fields (`src/param_registry.zig`). Duck typing, checked by the
compiler.

## 1.13 `@compileError`: the compiler as policy enforcer

Second "Zig can do THIS" moment. You can make compilation fail, on purpose,
with your own message — and because comptime code is ordinary code, the
*conditions* for failing can be arbitrarily smart. Fucina uses this for
misuse ("duplicate tensor tag", `src/tags.zig:161`; "too many tensor tags")
but also for **architecture**. From `src/fucina.zig:33-42`:

```zig
comptime {
    // Anti-regression guard: re-exporting the raw tensor type at the
    // PUBLIC ROOT is a COMPILE ERROR. This fires on any build that analyzes the
    // module root (every test/example/tool), not just `zig build test`. `internal`
    // and `bench_raw` are unaffected (this only inspects the root's own decls).
    if (@hasDecl(@This(), "RawTensor")) @compileError(
        "fucina.RawTensor must not be exported at the public root; raw tensors are internal. " ++
            "Use fucina.internal.RawTensor (in-tree raw naming) or bench_raw.RawTensor (microbench).",
    );
}
```

Context: underneath the tagged `Tensor` facade lives a raw, untagged f32
tensor, and the design decision — recorded in the comment right above this
guard — is that it stays internal: "the no-grad `Tensor` facade has
negligible forward overhead, so model/example code carries
`fucina.Tensor(spec)` end-to-end" (`src/fucina.zig:28-32`; a dedicated
microbench, `zig build bench-facade`, keeps that claim measurable). But a
comment is a wish. This `comptime` block is a *law*: it reflects on the
module's own declarations (`@hasDecl(@This(), ...)` — a file is a struct,
remember), and if anyone ever adds `pub const RawTensor = ...` to the public
root, every build of every test, example, and tool fails with a written
explanation of the policy and where to go instead.

API policy, enforced by the type checker, with a custom error message.
[Chapter 16](16-the-craft.md) collects more of these self-enforcing
disciplines; this is the pattern at its purest.

## 1.14 `@Vector`: SIMD as a type (a teaser)

One more feature, previewed now and studied properly in
[Chapter 06](06-going-fast-on-cpus.md): `@Vector(N, T)` is a SIMD vector as a
first-class type. Arithmetic on it (`+`, `*`, comparisons, `@splat`,
`@reduce`) compiles to the target's vector instructions — NEON on Apple
Silicon, AVX on x86 — with no intrinsics and no per-ISA source. The kernel
layer picks its width by asking the compiler what the target likes
(`src/backend/vector/common.zig:24-25`):

```zig
pub const vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;
pub const Vf32 = @Vector(vector_len, f32);
```

and then writes kernels once, generically over that width — here is the heart
of vector addition (`src/backend/vector/primitives.zig`, the simple tail of
`vecAdd`; the full kernel unrolls this 4×):

```zig
while (i + vector_len <= z.len) : (i += vector_len) {
    const xv: Vf32 = x[i..][0..vector_len].*;
    const yv: Vf32 = y[i..][0..vector_len].*;
    z[i..][0..vector_len].* = xv + yv;
}
while (i < z.len) : (i += 1) z[i] = x[i] + y[i];
```

`x[i..][0..vector_len].*` — slice, take a comptime-length sub-array, load it
as a vector value. One source; the compiler emits whatever the machine has.
That is all for now — how these become GEMM kernels, how comptime
CPU-feature gates compile unused ISA arms out of the binary entirely, and
how the scalar backend keeps all of it honest, is Chapter 06's whole story.

## 1.15 `test` blocks: the tests live with the code

You have been reading tests all chapter, because in Zig a test is a language
construct, not a framework:

```zig
test "name" {
    try std.testing.expect(...);
}
```

`test` blocks compile only under the test runner (`zig test file.zig`, or
`zig build test`); `std.testing` provides the assertions; returning
`error.SkipZigTest` skips cleanly — Fucina's model-asset-dependent suites do
exactly that when assets are missing, so `zig build test` is always green on
a fresh clone (`docs/REFERENCE.md` §2.7).

Fucina layers a simple convention on top (`docs/REFERENCE.md` §2.7): tests
live in **sibling files** — `exec.zig` has `exec_tests.zig`, 143 such files
across the tree — and the production file pulls its sibling in with a
forwarding stanza:

```zig
test {
    _ = @import("exec_tests.zig");
}
```

An anonymous test block that merely *references* the test file is enough to
compile it in. Module roots forward everything — `src/fucina.zig` ends with
`test { _ = dtype; _ = exec; ... }` (`src/fucina.zig:169-189`) — so nine test
roots reach every test in the repository.

Then the convention eats its own documentation: `zig build snippet-check`
extracts every runnable snippet from `docs/REFERENCE.md` (any fenced block
with a named `test`) and runs it against the real modules, as a CI step
(`docs/REFERENCE.md` §2.7). The first program in §1.2 is not an illustration
that *resembles* the library; it is a test that *runs against* it on every
push and pull request. Docs that cannot rot. This course borrows the ethic:
every fresh snippet above was compiled with `zig test` before it was pasted.

Why a tensor library needs this: numerical code fails quietly. The only
defense is tests so cheap to write and run that they are everywhere — same
file, same language, same build graph, leak checking free of charge (§1.3).

## 1.16 The build system is a Zig program

There is no Makefile, no CMake, no shell scripts taped together. `build.zig`
exports one function — `pub fn build(b: *std.Build) void` (`build.zig:7`) —
and that *program* declares modules, executables, options, and test steps
(you have been using it: `zig build test`, `zig build spirals`). The part
that matters downstream: **project options become compile-time constants**.
`build.zig` declares its option enums at file scope (`build.zig:3-5`):

```zig
const BackendKind = enum { scalar, native, cpu };
const BlasKind = enum { none, accelerate, openblas, mkl, blis, nvpl, blas };
const GpuKind = enum { none, metal, cuda };
```

collects `-D` flags, and bakes them into a generated module that source files
import (`build.zig:76-91`, abridged):

```zig
const options = b.addOptions();
options.addOption(BackendKind, "backend_kind", backend_kind);
options.addOption(BlasKind, "blas_kind", blas_kind);
options.addOption(usize, "max_threads", max_threads);
// ...
const module = b.addModule("fucina", .{
    .root_source_file = b.path("src/fucina.zig"),
    .target = target,
    .optimize = optimize,
});
module.addOptions("build_options", options);
```

Now `@import("build_options").backend_kind` is a comptime value — which is
how §1.9's backend switch could run in the compiler. The documented
consequence: "backend dispatch is compiled away, and unused kernel arms are
not in the binary" (`docs/REFERENCE.md` §2.2). Configuration is not read at
startup; it is a property of the binary.

The options you need this course (full table: `docs/REFERENCE.md` §2.2):

- **`-Doptimize=ReleaseFast`** — the one to remember. "Build with
  `ReleaseFast` whenever speed matters (Debug is 10–50× slower); validate in
  Debug/ReleaseSafe, bench in ReleaseFast" (`docs/REFERENCE.md` §2.2). The
  companion trap: ReleaseFast *drops safety checks* — "A kernel that only
  behaves because Debug catches it is broken; prove invariants, don't rely on
  checks as logic" (`AGENTS.md`, Zig 0.16 notes).
- **`-Dbackend=native|scalar`** — the two CPU backends of §1.9: `native`
  (default, Zig SIMD kernels) and `scalar` (the reference oracle; `cpu` is a
  deprecated alias). BLAS (`-Dblas=...`) is not a third backend — it is an
  optional GEMM *provider* backing the native backend's large-matmul arms;
  likewise `-Dgpu=metal|cuda` is a GPU *offload seam*, not a backend
  ([Chapter 06](06-going-fast-on-cpus.md)).
- **`-Dmax-threads=N`** — a comptime ceiling on the worker team (default 8;
  many-core machines must raise it at build time).

Misconfiguration is a **build-time panic**, not a runtime error:
`-Dblas=accelerate` off macOS, or `-Dmax-threads=0`, panics inside `build()`
with a message (`docs/REFERENCE.md` §2.2).

Finally, the caveat that bites people who deploy: **CPU targeting is native
by default**. With no `-Dtarget`, Zig compiles for the *exact* CPU of the
building machine — full feature set, like `-march=native` — and Fucina's
comptime feature gates compile in the matching fast kernels (NEON/dotprod on
Apple Silicon, AVX2/AVX-VNNI on modern x86). A bare `-Dtarget=...` drops to
that architecture's **baseline** and "silently loses the fast kernels"
(`AGENTS.md`, build options) — pin `-Dcpu` to a model as well, e.g.
`-Dtarget=x86_64-linux -Dcpu=x86_64_v3`. Two rules: build on the machine
that will run the binary, or name its CPU explicitly.

> **ML note** — this replaces Python-land's "wheels compiled for which AVX
> level?" problem. Your binary is specialized to your machine by default,
> and the specialization is visible in two flags.

## 1.17 Zig 0.16, not the Zig on the internet

The repo documents the version deltas that most often trip people whose Zig
knowledge predates 0.16 (`AGENTS.md`, "Zig 0.16 notes"; full reference at
[ziglang.org/documentation/0.16.0](https://ziglang.org/documentation/0.16.0/)):

- **`usingnamespace` was removed.** Compose with explicit `pub const`
  re-exports (as `src/fucina.zig` does, §1.7).
- **`@splat(scalar)`** takes only the scalar; the vector length is inferred
  from the result type (you saw this in §1.14).
- **Overflow builtins return tuples**: `const r, const ov = @addWithOverflow(a, b);`.
- **Casts infer their destination** from context (`@intCast`, `@ptrCast`,
  `@enumFromInt`, ...); use `@as(T, x)` to state a target type explicitly.
- **`main` takes `std.process.Init`**, and I/O goes through the new `std.Io`
  writer API. From `examples/spirals.zig:327-339`:

```zig
pub fn main(init: std.process.Init) !void {
    // ... allocator setup as in §1.3 ...
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    defer stdout.flush() catch {};
```

  An explicit buffer, a writer over it, and `defer stdout.flush() catch {}` —
  best-effort cleanup that swallows the error, because there is nowhere
  sensible to report a failed flush on exit.

- One that bit this very chapter's snippets: inside a `comptime` block, a
  plain `for` already unrolls — 0.16 rejects a redundant `inline` keyword
  there.

The rule from the chapter's start, restated as a habit: when fresh Zig
disagrees with something you read elsewhere, imitate this repo — then run
`zig test`, because the compiler is the only tutorial never out of date.

## What you now know

- Fucina builds with exactly Zig 0.16.0; `zig build test` runs nine test
  roots with no model assets, and a leaked byte fails the build.
- Memory is a parameter: allocators are explicit, `defer`/`errdefer` pair
  every acquire with a visible release, and `deinit` ends in
  `self.* = undefined` as a Debug tripwire.
- Errors are values in closed sets (`TensorError`); `try` makes every failure
  path visible at the call site.
- Arrays carry comptime length, slices are runtime views, and `[]T` vs
  `[]const T` turns the own/borrow doctrine into type checking — kernel
  signatures like `vecAdd(z: []f32, x: []const f32, ...)` *are* the contract.
- Structs + methods + files-as-namespaces are the whole module system; a
  model is just a struct of tensors. Optionals make "has a gradient or not"
  (`grad_state: ?*GradState`) a type fact; exhaustive `switch` makes
  dtype/backend dispatch un-forgettable.
- `comptime` runs ordinary Zig in the compiler — values, then types, then
  functions returning types: `Tensor(.{ .batch, .in })` is a comptime call
  and the tags compile away to nothing. Tag-level mistakes fail compilation;
  axis *sizes* stay runtime.
- `pub const Tag = @TypeOf(.tag)` and the `@compileError` RawTensor guard:
  the compiler as metaprogramming substrate and as policy enforcer.
- `@Vector` gives portable SIMD as a type (Chapter 06 goes deep); `test`
  blocks live beside the code, and even the reference manual's snippets run
  in CI.
- The build system is a Zig program; `-D` options become comptime constants;
  `ReleaseFast` when speed matters (Debug is 10–50× slower); two CPU backends
  (`scalar` oracle, `native` fast); never ship a bare `-Dtarget` binary.

## Explore the source

- `docs/REFERENCE.md` §1–§2 — the mental model and build/toolchain reference
  this chapter quoted throughout; §1.4 is the first program.
- `src/fucina.zig` — the public root: re-exports, the `internal` seam, the
  `@compileError` guard, the forwarding test stanza.
- `src/tensor.zig` (first ~80 lines) — `TensorError`, `Shape`,
  `isContiguous`: arrays, slices, error unions, `inline while` at work.
- `src/tags.zig` — pure comptime programming: `Tag`, `tagEqual`, `tagIndex`,
  the comptime bubble sort. Readable in isolation, no tensors required.
- `src/ag/tensor.zig` (lines ~150–270) — the `Tensor` type constructor,
  `grad_state: ?*GradState`, ownership doc-comments, `errdefer` in `variable`.
- `examples/spirals.zig` — a complete train/checkpoint/infer program in one
  file; Chapter 08 dissects it line by line.
- `build.zig` — the build as a program: option enums, panics as validation,
  `addOptions`.

## Exercises

1. **Break the leak checker.** Copy the `sumOfSquares` snippet from §1.3 into
   a file, delete the `defer allocator.free(buf);` line, and run `zig test`.
   Read the diagnostic — the testing allocator reports the allocation's stack
   trace. Put the `defer` back; now make the function *fail* after allocating
   and confirm the `defer` still frees on the error path.
2. **Grow `Shape`.** Extend the course `Shape` from §1.7 with a
   `fn equal(a: *const Shape, b: *const Shape) bool` method and a test, then
   compare with the real one in `src/tensor.zig:21-53`. What does
   `initStrides` allow that `init` rejects, and why might strides
   legitimately contain a zero?
3. **Provoke the tag system.** In a test that imports the library (easiest:
   temporarily add one inside `src/fucina_tests.zig` — it already imports the
   root — and run `zig build test-fucina`), construct `x` and `w` as in the
   first program, then try
   `x.dot(&ctx, &w, .out)` — an axis `x` does not have. Find the
   `@compileError` call in `src/tags.zig` that produced the message.
4. **A generic of your own.** Write
   `fn Pair(comptime A: type, comptime B: type) type` with fields
   `first: A`, `second: B` and a `swap` method returning `Pair(B, A)`;
   compile-check it with `zig test`. Then read `TopKResult` in
   `src/ag/tensor.zig:167-183` — a real two-field generic whose second
   field's *type* is computed from the first's spec.
5. **(Harder) Read one real comptime function end to end.** Annotate
   `dotResultTags` in `src/tags.zig`: which values exist only at comptime,
   where `inline for` unrolls, and what the returned array's *length* being
   computed by another comptime function (`dotResultLen`) implies about how
   Zig types depend on values. Predict the result tags for
   `(.{ .batch, .in }) · (.{ .in, .out })` contracting `.in`, then check
   against the first program's comment.

---

[Previous: Introduction — why deep learning in Zig?](00-introduction.md) ·
[Next: Just enough machine learning](02-just-enough-ml.md)
