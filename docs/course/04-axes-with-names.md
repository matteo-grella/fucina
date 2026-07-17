# Chapter 04 — Axes with names: types that know their shape

*Part II — The tensor core*

In [Chapter 3](03-tensors-from-scratch.md) we built the raw tensor: a flat
refcounted buffer plus a shape and strides, all of it runtime data. It
works, and it has the problem every tensor library has: axis **0** of one
tensor must line up with axis **1** of another, and nothing in the program
text says which is which. This chapter fixes that with the most Zig idea in
all of Fucina: axis *names* carried in the *type*.
`Tensor(.{ .batch, .in })` is a different type from
`Tensor(.{ .in, .batch })`, contraction happens by name, the result's names
are computed while the compiler runs, and a contraction over an axis an
operand doesn't have is a compile error — the program that would compute
the wrong shape is not a program.

This chapter is also the course's real introduction to `comptime`.
[Chapter 1](01-just-enough-zig.md) showed the mechanics; here you watch
comptime carry a whole subsystem: values that exist only at compile time,
functions that return types, array lengths computed by other functions, and
`@compileError` as a domain-specific error channel.

## 4.1 The shape-bug problem

> **ML note** — Ask anyone who maintains model code what their most common
> bug is. It is not a wrong formula; it is a *shape bug*. The nasty ones
> don't crash — a transposed weight matrix multiplied the wrong way round
> often produces tensors of a plausible shape full of wrong numbers, and
> the only symptom is a loss curve that refuses to fall. Positional APIs
> make this easy: `matmul(a, b)` contracts "the last axis of `a` with the
> first of `b`", and it is entirely on you to remember what those axes
> *mean* everywhere in the program.

Fucina's answer is that an axis is identified by a **tag** — a name like
`.batch`, `.seq`, `.d_model` — and the name lives in the tensor's *type*.
Here is the destination, the README's own condensed example (README.md:24–41;
the full-size version is `examples/spirals/main.zig:133–142`, and the same
pattern runs the production trainer in `src/llm/qwen3/train.zig`):

```zig
const Model = struct {
    w1: Tensor(.{ .h1, .in }),
    b1: Tensor(.{.h1}),
    w2: Tensor(.{ .class, .h1 }),
    b2: Tensor(.{.class}),
};

fn forward(ctx: *ExecContext, m: *const Model, x: *const Tensor(.{ .batch, .in })) !Tensor(.{ .batch, .class }) {
    var z1 = try x.dot(ctx, &m.w1, .in); // contract .in -> .{ .batch, .h1 }
    defer z1.deinit();
    var s1 = try z1.add(ctx, &m.b1);
    defer s1.deinit();
    var a1 = try s1.tanh(ctx);
    defer a1.deinit();
    var z2 = try a1.dot(ctx, &m.w2, .h1); // contract .h1 -> .{ .batch, .class }
    defer z2.deinit();
    return try z2.add(ctx, &m.b2);
}
```

Read the signature alone: data goes in as `(batch, in)` and comes out as
`(batch, class)`; each `dot` says *which* axis it consumes. The README
frames it with a deliberately old-fashioned analogy (README.md:56–60):

> The shape discipline lives in the program text, the way it did in Fortran —
> `real A(n,m)` told you the rank before you read a single loop. Here the
> signature `x: Tensor(.{ .batch, .in }) -> Tensor(.{ .batch, .class })` lets
> you check the algorithm against the math by eye, and Zig's comptime makes it
> free: the tags exist only in the type system and compile away entirely.

Every axis-level decision in the library — broadcasting, contraction,
reduction, permutation — is "made by tag identity, never by axis position at
the call site" (docs/REFERENCE.md, §7 opening). The approach is credited:
"the tagged-tensor approach (axis tags carried in the type, operands aligned
by name) was inspired by [ZML](https://github.com/zml/zml)"
(README.md:327–328). What Fucina adds is the particular split this chapter
teaches: names and ranks at compile time, sizes and layout at runtime.

Two internal modules implement the machinery: `src/tags.zig` (a pure
comptime tuple algebra) and `src/tagged.zig` (runtime ops directed by
comptime tags). **Neither is public API** — they are not re-exported at the
module root; user code consumes all of this through `Tensor` methods, and
the autograd VJPs call the same library on raw gradients (docs/REFERENCE.md
§7). We will read them anyway — they are the best comptime tutorial in the
tree — and build a miniature version first, as always. One caveat for
everything that follows: Fucina's public API is explicitly young (no
semver, no package manifest), so every signature in this chapter is
*today's code*, not a frozen contract.

## 4.2 A name you can hold: enum literals and `Tag`

The entire type system rests on two lines (src/tags.zig:4–5):

```zig
pub const Tag = @TypeOf(.tag);
pub const inserted_axis = std.math.maxInt(usize);
```

Ignore the second line for now (it returns in §4.5); the first deserves a slow look.

> **Zig note** — `.tag` with no enum type in sight is an **enum literal**: a
> value that says "the name `tag`" without belonging to any declared enum.
> Normally you meet enum literals as shorthand — `const c: Color = .red;`
> coerces the literal to `Color`. But an *uncoerced* enum literal is a value
> too, and it has a type: `@TypeOf(.tag)` is Zig's anonymous enum-literal
> type. That type is **comptime-only** — you cannot store an enum literal in
> a runtime variable, because there is no runtime representation for "some
> name". Which is exactly what we want: names that exist while the compiler
> runs and are gone from the binary.

So a `Tag` is any identifier you like — `.batch`, `.qkv`, `._0` — with no
registry of valid names. "Two tags are equal iff their spellings are equal"
(docs/REFERENCE.md §7.1); equality is a compile-time string comparison of
`@tagName` (src/tags.zig:190–201):

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

A `while` loop over the characters of a string, running inside the
compiler: the `comptime blk:` label forces the computation to compile time,
and the result is a plain `bool` the caller can branch on — also at compile
time. Names carry no built-in meaning; there is no blessed `.batch`
semantics anywhere in the library — "`._0`…`._7` are ordinary tags that
happen to be generated for rank specs" (docs/REFERENCE.md §7.1). Identity
is all a tag is.

Let's start the chapter's build-it-yourself layer. All course code in this
chapter lives in one scratch file and passes `zig test` with Zig 0.16.0
(pure `std`, no Fucina imports — this is *course code*, not repo code):

```zig
const std = @import("std");

pub const Tag = @TypeOf(.tag);

pub fn tagEqual(comptime a: anytype, comptime b: anytype) bool {
    return comptime std.mem.eql(u8, @tagName(a), @tagName(b));
}

pub fn tagIndex(comptime tags: anytype, comptime tag: anytype) ?usize {
    inline for (tags, 0..) |candidate, i| {
        if (comptime tagEqual(candidate, tag)) return i;
    }
    return null;
}

pub fn tagIndexOrCompileError(comptime tags: anytype, comptime tag: anytype) usize {
    inline for (tags, 0..) |candidate, i| {
        if (comptime tagEqual(candidate, tag)) return i;
    }
    @compileError("tensor tag not found");
}

pub fn validateUniqueTags(comptime tags: anytype) void {
    inline for (0..tags.len) |i| {
        inline for ((i + 1)..tags.len) |j| {
            if (comptime tagEqual(tags[i], tags[j])) @compileError("duplicate tensor tag");
        }
    }
}

test "tags are comptime values with spelling equality" {
    comptime {
        std.debug.assert(tagEqual(.batch, .batch));
        std.debug.assert(!tagEqual(.batch, .seq));
        std.debug.assert(tagIndex(.{ .batch, .seq, .d }, .d) == 2);
        std.debug.assert(tagIndex(.{ .batch, .seq, .d }, .channel) == null);
    }
}
```

(Our `tagEqual` cheats with `std.mem.eql`, which also works at comptime; the
repo spells out the loop.) Three Zig ideas are doing the work:

- **`anytype` + `comptime`**: `tags` is an anonymous *tuple* of enum
  literals like `.{ .batch, .seq, .d }` — comptime-only values, so these
  parameters could not be runtime even if we wanted them to be.
- **`inline for`**: a loop the compiler unrolls — the only way to iterate a
  tuple, whose elements may differ in type.
- **`@compileError`**: our error channel. `tagIndex` returns `?usize` for
  callers that can handle absence; `tagIndexOrCompileError` is for callers
  whose *semantics require* presence. The real library's message — `"tensor
  tag not found"` (src/tags.zig:187) — will appear verbatim in §4.7 when we
  break a real contraction.

`validateUniqueTags` (repo version at src/tags.zig:158–164) enforces the
invariant that a tag tuple never repeats a name — `"duplicate tensor tag"` —
with an O(n²) pairwise check, free because it runs at compile time.

## 4.3 Functions that return types: `Tensor(spec)`

> **Zig note** — Zig has no generics syntax. Types are comptime *values*,
> so a "generic type" is an ordinary function that takes comptime parameters
> and returns a `type`. `std.ArrayList(u8)` is a function call; so is
> `Tensor(.{ .batch, .in })`. The compiler memoizes these calls — same
> argument values, same returned type.

Here is the real constructor, elided to its skeleton
(src/ag/tensor.zig:185–211):

```zig
pub fn Tensor(comptime tags_spec: anytype) type {
    const tensor_dtype = dtypeFromSpec(tags_spec);
    if (comptime tensor_dtype == .f32) return FloatTensor(tags_spec);
    if (comptime dtype_mod.isBlockQuantized(tensor_dtype)) return QuantizedConstantTensor(tags_spec, tensor_dtype);
    return TypedConstantTensor(tags_spec, tensor_dtype);
}

fn FloatTensor(comptime tags_spec: anytype) type {
    const tags = normalizeTags(tags_spec);
    comptime validateUniqueTags(tags);
    const tag_rank = tags.len;
    if (tag_rank > tensor_mod.max_rank) @compileError("too many tensor tags");

    return struct {
        pub const axis_tags = tags;
        pub const tag_count = tag_rank;
        pub const tensor_rank = rawRank(tag_rank);
        pub const dtype = DType.f32;

        value: RawTensor,
        grad_state: ?*GradState = null,
        scope_owned: bool = false,
        // ... every method of the f32 tensor follows
    };
}
```

Look at the *fields*: a raw tensor (Chapter 3's value), an optional gradient
pointer (Chapter 7's business), and a bool. No tag array. The tags appear
only as `pub const` declarations — facts about the *type*, with zero
per-instance footprint. This is the mechanical substance behind "the tags
compile away entirely", and §4.10 will squeeze the last drop out of it.

The dtype dispatch adds a design point worth noticing: each branch's method
set is decided at compile time, so calling an operation a dtype doesn't
support "is a compile error, never a runtime failure" (docs/REFERENCE.md
§3.2) — an i64 tensor simply *has no* `tanh` method.
[Chapter 5](05-the-operation-library.md) tours what each branch can do.

### Spec forms and normalization

`Tensor(spec)` accepts five spellings, all normalized by `src/tags.zig`
(table from docs/REFERENCE.md §3.1):

| Spec form | Example | Meaning |
|---|---|---|
| Named tag tuple | `Tensor(.{ .batch, .in })` | rank = tuple length, dtype `.f32` |
| Numeric rank | `Tensor(2)` | rank-2 f32 with auto tags `._0, ._1` |
| dtype + tags | `Tensor(.{ .dtype = .f16, .tags = .{ .seq, .d } })` | typed, named axes |
| dtype + rank | `Tensor(.{ .dtype = .i64, .rank = 2 })` | typed, auto tags |
| Scalar | `Tensor(.{})` | zero tags; stored as raw rank-1 shape `{1}` |

The normalizer is comptime duck typing (src/tags.zig:85–97):

```zig
pub fn normalizeTags(comptime tags_spec: anytype) [tagSpecLen(tags_spec)]Tag {
    if (comptime isTensorSpec(tags_spec)) {
        const Spec = @TypeOf(tags_spec);
        if (@hasField(Spec, "tags")) return normalizeTags(tags_spec.tags);
        if (@hasField(Spec, "rank")) return autoTags(rankFromSpec(tags_spec.rank));
        @compileError("tensor dtype specs must include tags or rank");
    }
    if (comptime isRankSpec(tags_spec)) return autoTags(rankFromSpec(tags_spec));

    var out: [tagSpecLen(tags_spec)]Tag = undefined;
    inline for (0..out.len) |i| out[i] = tags_spec[i];
    return out;
}
```

Two teaching points hide here. First, **comptime introspection**:
`isTensorSpec` switches on `@typeInfo(@TypeOf(tags_spec))` to recognize a
non-tuple struct, `@hasField` asks what it carries, and recursion flattens
three input shapes to one form. Second, the **"Len twin" pattern** in the
return type: `[tagSpecLen(tags_spec)]Tag`. Zig array types carry their
length, and you cannot return an array of unknown length — so every
tuple-rewriting function in `src/tags.zig` comes as a pair, one computing
the length, one filling the array, the second naming the first in its own
return type. Watch for it throughout: `pointwiseResultLen` /
`pointwiseResultTags`, `dotResultLen` / `dotResultTags`, and friends.

`max_rank` is 8 (src/tensor.zig:9). Every result-tag computation enforces it
with `@compileError("too many tensor tags")` — a *rank overflow* is a
tag-level violation and fails at compile time, like every other tag-level
violation.

### Type identity: one type per meaning, not per spelling

Now the crucial question. If `Tensor` is a function, and `Tensor(2)` and
`Tensor(.{ ._0, ._1 })` are different calls with different argument values —
are the resulting types different? If they were, spelling would fragment the
API: a helper that returns `Tensor(2)` couldn't hand its result to a
function expecting `Tensor(.{ ._0, ._1 })`.

The documented contract says no fragmentation: "Specs that normalize to the
same (dtype, tag list) produce the *same* type" (docs/REFERENCE.md §3.1).
It is pinned by a machine-verified snippet — every runnable snippet in
REFERENCE.md is compiled and run against the tree by the `zig build
snippet-check` CI gate — and this one asserts type equality directly
(docs/REFERENCE.md §3.1):

```zig
test "Tensor spec forms and comptime introspection" {
    const A = fucina.Tensor(.{ .batch, .in }); // named tags, dtype defaults to f32
    const B = fucina.Tensor(2); // numeric rank: axes tagged ._0, ._1
    const C = fucina.Tensor(.{ .dtype = .i64, .rank = 2 }); // typed, auto tags
    const D = fucina.Tensor(.{ .dtype = .f16, .tags = .{ .seq, .d } }); // typed, named tags
    const S = fucina.Tensor(.{}); // scalar
    comptime {
        std.debug.assert(A.dtype == .f32 and A.tag_count == 2 and A.tensor_rank == 2);
        std.debug.assert(B.axis_tags[0] == ._0 and B.axis_tags[1] == ._1);
        std.debug.assert(C.dtype == .i64 and D.dtype == .f16);
        std.debug.assert(S.tag_count == 0 and S.tensor_rank == 1); // scalars store rank-1 [1]
        // Specs that normalize to the same (dtype, tags) are the SAME type:
        std.debug.assert(B == fucina.Tensor(.{ ._0, ._1 }));
        std.debug.assert(A == fucina.Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } }));
    }
}
```

Notice *where* Fucina normalizes: `FloatTensor` computes
`const tags = normalizeTags(tags_spec);` **before** `return struct { ... }`,
so every comptime value the struct captures is already canonical. Spellings
that mean the same thing converge before the type is declared; with Zig
0.16.0, the returned types are then equal, as the snippet gate verifies on
every CI run. That is the contract this course states — observed,
test-pinned behavior plus the documented rule — and we won't speculate
about compiler internals beyond it. The flip side needs no subtlety at all:
`Tensor(.{ .batch, .in })` and `Tensor(.{ .in, .batch })` normalize to
*different* tag lists, so they are different types, and passing one where
the other is expected is an ordinary type error.

One more corner from the table: `Tensor(.{})` has `tag_count == 0` but
`tensor_rank == 1` — Fucina has no rank-0 tensors, so a scalar is stored as
a rank-1, one-element tensor of shape `{1}`, bridged by `rawRank(0) == 1`
(src/tags.zig:203–205; the no-zero-size/no-rank-0 stance is documented in
REFERENCE.md §3.1). Full reductions like `sumAll` return exactly this type.

### The mini version

Our course-code equivalent — compile-checked, tests passing:

```zig
/// The mini type constructor: a read-only f32 view whose axis NAMES live in
/// the type. Sizes and strides stay runtime values.
pub fn View(comptime tags_spec: anytype) type {
    const tags = normalizeTags(tags_spec); // normalize BEFORE declaring the struct
    comptime validateUniqueTags(tags);

    return struct {
        pub const axis_tags = tags;
        pub const rank = tags.len;

        data: []const f32,
        shape: [rank]usize,
        strides: [rank]usize,

        const Self = @This();

        /// Wrap a dense row-major buffer (strides computed as in Chapter 3).
        pub fn fromSlice(data: []const f32, shape: [rank]usize) Self { ... }

        pub fn dim(self: Self, comptime tag: Tag) usize {
            return self.shape[comptime tagIndexOrCompileError(axis_tags, tag)];
        }

        pub fn at(self: Self, idx: [rank]usize) f32 {
            var off: usize = 0;
            for (idx, self.strides) |i, s| off += i * s;
            return self.data[off];
        }
    };
}
```

(Our `normalizeTags` mirrors the repo version at toy scale: it accepts a
tag tuple or an integer rank — `View(2)` auto-tags as `._0, ._1` — and
returns a plain `[N]Tag` array; the full scratch file has it and the
elided `fromSlice` body.) The tests that make the two identity claims
concrete, both passing with Zig 0.16.0:

```zig
test "same spelling, same type; different order, different type" {
    comptime {
        std.debug.assert(View(.{ .batch, .in }) == View(.{ .batch, .in }));
        std.debug.assert(View(.{ .batch, .in }) != View(.{ .in, .batch }));
        // Normalization converges the spellings: rank 2 == explicit auto tags.
        std.debug.assert(View(2) == View(.{ ._0, ._1 }));
    }
}

test "instances carry no tags at runtime" {
    // The struct stores data + shape + strides; the names are type-level only.
    const V = View(.{ .batch, .d });
    try std.testing.expectEqual(
        @sizeOf([]const f32) + 2 * @sizeOf([2]usize),
        @sizeOf(V),
    );
}
```

The second test is the "compiles away" claim measured with a ruler: a
rank-2 view is exactly one slice and two `[2]usize` arrays, names included
free of charge.

`dim` is worth a pause: the *lookup* — which axis holds `.d`? — happens at
compile time (`comptime tagIndexOrCompileError(...)` folds to a constant
index), while the *answer* — how big is that axis? — is a runtime load. That
one line is the whole philosophy of the subsystem.

## 4.4 Honesty first: names at comptime, sizes at runtime

Before the payoff sections, the boundary — overselling this type system is
the easiest mistake to make, and the docs are careful not to. "Rank, tags,
and dtype are comptime; sizes are runtime" (docs/REFERENCE.md §3.1). The
type knows *what the axes are called* and *how many there are*, not *how
long* they are. There are two failure worlds, and you must keep them
straight:

| Violation | Example | Fails |
|---|---|---|
| Contract tag missing from an operand | `a.dot(ctx, &b, .k)` where `b` has no `.k` | **compile time** |
| Duplicate tag in a spec | `Tensor(.{ .d, .d })` | **compile time** |
| Rank overflow (> 8 tags) | pointwise union of 9 distinct tags | **compile time** |
| Permutation target with wrong tag set | `permuteTo(.{ .a, .z })` on `{ .a, .b }` | **compile time** |
| Same tag, different sizes | `.d` of 3 added to `.d` of 2 | **runtime**, `error.ShapeMismatch` |
| Split factors don't multiply to the axis size | split 6 into `{ 4, 2 }` | **runtime**, `error.InvalidShape` |
| Merge over stride-incompatible layout | merge across a transposed view | **runtime**, `error.UnsupportedView` |

The top half is the tag algebra: "Every function here runs at compile time
and violations are compile errors" (docs/REFERENCE.md §7 on `src/tags.zig`).
The bottom half is the size world, where errors are ordinary recoverable Zig
errors — "All failures are recoverable Zig errors; nothing in this layer
panics" (docs/REFERENCE.md §7.5, which also tabulates the runtime errors).

Here is the runtime half in a machine-verified snippet (docs/REFERENCE.md
§7.7, context setup elided). Two tensors agree perfectly on names — both are
`{.d}` — and disagree on size, 3 versus 2:

```zig
test "incompatible dims fail with ShapeMismatch" {
    var a = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 10, 20 });
    defer b.deinit();

    try std.testing.expectError(error.ShapeMismatch, a.add(&ctx, &b));
    try std.testing.expectError(error.ShapeMismatch, a.dot(&ctx, &b, .d));
}
```

Both calls *compile* — the names are fine — and both fail at runtime. So
what did the type system buy, if it can't see a 3 meeting a 2?

It bought the errors that positional APIs turn into *silent wrong answers*.
A size mismatch on a correctly-named axis is the loud kind of bug: a named
error, at the call site, the first time the code runs. The quiet kind —
contracting the wrong axis, adding along a transposed pair of equal-sized
dimensions — is precisely the kind where sizes match and *meaning* doesn't.
Names catch the quiet kind, before the program exists. That trade is the
design: extents are data, meanings are types.

> **ML note** — The split mirrors how real model code lives. `d_model =
> 4096` is a config value read from a GGUF file at startup
> ([Chapter 11](11-model-files-and-quantization.md)); no type system should
> need recompiling per model size. But "queries contract with keys over
> `.head_dim`" is a fact about the *architecture*, identical for every
> checkpoint — exactly the kind of fact you want written once, in a type.

## 4.5 Alignment by name: views that lie about their shape

Everything from here on is one recipe: *comptime functions decide what the
axes mean; runtime code moves sizes and strides accordingly*. The first
application is the subsystem's workhorse — aligning one tensor's axes to
another's order.

A view ([Chapter 3](03-tensors-from-scratch.md)) is a shape and strides
over a shared buffer; the tag layer adds that you never permute *by
position*, you align *to a target tag order*. Here is the real
implementation, short enough to read whole (src/tagged.zig:330–362,
`alignTensorToOf`; `alignTensorTo` at :324 is its f32 wrapper):

```zig
pub fn alignTensorToOf(
    comptime tensor_dtype: DType,
    comptime source_tags: anytype,
    source: *const tensor_mod.TensorOf(tensor_dtype),
    comptime target_tags: anytype,
) !tensor_mod.TensorOf(tensor_dtype) {
    comptime {
        validateUniqueTags(target_tags);
        if (target_tags.len > tensor_mod.max_rank) @compileError("too many tensor tags");
        for (source_tags) |tag| {
            if (tagIndex(target_tags, tag) == null) @compileError("target tags must include all source tags");
        }
    }
    try validateTensorRankOf(tensor_dtype, source_tags, source);

    // (scalar-target fast path elided)

    var shape: [target_tags.len]usize = undefined;
    var strides: [target_tags.len]usize = undefined;
    inline for (target_tags, 0..) |target_tag, i| {
        if (tagIndex(source_tags, target_tag)) |source_i| {
            shape[i] = source.shape.at(source_i);
            strides[i] = source.strides.at(source_i);
        } else {
            shape[i] = 1;
            strides[i] = 0;
        }
    }

    return source.viewWithStrides(shape[0..], strides[0..]);
}
```

The structure is the recipe made visible: a `comptime { ... }` block up top
does the tag-level checks (target unique, within rank bounds, a **superset**
of the source — all compile errors); then a runtime loop whose *iteration
structure* is comptime — `inline for` over the target tags, `tagIndex`
folded to a constant per iteration — fills two plain runtime arrays. A
target tag the source has copies its size and stride; a target tag the
source *lacks* gets **size 1, stride 0**.

That size-1/stride-0 axis is the most important trick in the chapter. A
stride of 0 means "advancing along this axis doesn't move through memory" —
the same element read again. It makes alignment (and, next section,
broadcasting) **zero-copy always** (docs/REFERENCE.md §7.6): no buffer is
allocated, no element is touched; we only describe the old buffer with new
geometry. (And here `inserted_axis`, the `maxInt(usize)` sentinel from
§4.2, earns its keep: the comptime axis-map helper `alignAxes` marks
injected axes with it, meaning exactly this size-1/stride-0 pair.)

Our mini version is a method on `View` — the same checks and the same loop,
minus the dtype generality, with the signature
`pub fn alignTo(self: Self, comptime target_spec: anytype) View(normalizeTags(target_spec))`
(full body in the compile-checked scratch file). Its test makes the
stride-0 lie visible:

```zig
test "alignTo reorders and injects phantom axes" {
    const x = View(.{ .batch, .d }).fromSlice(&.{ 1, 2, 3, 4, 5, 6 }, .{ 2, 3 });
    const y = x.alignTo(.{ .d, .batch, .channel });
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 1 }, &y.shape);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 0 }, &y.strides);
    // Same buffer, transposed traversal: y[d=2][batch=1][channel=0] == x[1][2].
    try std.testing.expectEqual(@as(f32, 6), y.at(.{ 2, 1, 0 }));
}
```

Note the return type: `View(normalizeTags(target_spec))` — the method's
result *type* is computed from its comptime argument, so downstream code is
typed by the new order automatically. Every view method on the real facade
has this shape: `alignTo`, `permuteTo`, `transpose`, `withTags` (relabel),
`insertAxis`, `squeeze`, `broadcastTo` all return
`Tensor(<some tag computation>)` (signatures in src/ag/tensor.zig:1042–1148).
`permuteTo` is `alignTo` plus the constraint that the target is a
permutation — same length, same membership (`validateSameTagSet`,
src/tags.zig:166–170) — so no axis can be injected.

The facade twin of our test is machine-verified in docs/REFERENCE.md §7.6:
the same `{2,3}` tensor, `x.alignTo(&ctx, .{ .d, .batch, .channel })`, the
same `{3,2,1}` shape, and a `copyTo` reading out the transposed traversal
`1, 4, 2, 5, 3, 6` from the untouched original buffer.

One ownership note, carried over from Chapter 3: view results share storage
with their source ("writing through one aliases the other") but are owned
values — the caller must `deinit` them, and buffer refcounting keeps the
view valid even after the source value is gone (docs/REFERENCE.md §7.5).

## 4.6 Pointwise: broadcasting by name

Now the first *result-tag computation* — the comptime function deciding
what type `x.add(ctx, &bias)` returns (src/tags.zig:343–366):

```zig
pub fn pointwiseResultTags(comptime left_tags: anytype, comptime right_tags: anytype) [pointwiseResultLen(left_tags, right_tags)]Tag {
    var out: [pointwiseResultLen(left_tags, right_tags)]Tag = undefined;
    var out_i: usize = 0;
    inline for (left_tags) |tag| {
        out[out_i] = tag;
        out_i += 1;
    }
    inline for (right_tags) |tag| {
        if (comptime tagIndex(left_tags, tag) == null) {
            out[out_i] = tag;
            out_i += 1;
        }
    }
    return out;
}
```

The result tags are the **union in operand order**: all left tags in left
order, then every right-only tag appended in right order. The Len twin
(`pointwiseResultLen`, same file) counts the union and carries the rank cap
— `@compileError("too many tensor tags")` past `max_rank`. And the facade
`add` puts the computation directly in its return type
(src/ag/tensor.zig:1155; `sub`/`mul`/`div`/`maximum`/`minimum` are
identical in shape):

```zig
pub fn add(self: *const Self, ctx: *ExecContext, other: anytype) !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags))
```

> **Zig note** — `other: anytype` accepts a value or a pointer;
> `TensorObject(@TypeOf(other))` recovers the tensor type either way,
> *inside the return type expression*. Reading Fucina signatures means
> getting comfortable with return types that are function calls.

Given the result tags, broadcasting is one rule applied per axis
(docs/REFERENCE.md §7.7): an operand missing the tag contributes size 1
(`alignTo`'s injected phantom axis); equal sizes pass through; a size of 1
stretches to the other's size — zero-stride, no copy; two unequal non-1
sizes return `TensorError.ShapeMismatch`.

Both operands are aligned to the result tags as views, stretched, and the
rank-matched kernel from [Chapter 5](05-the-operation-library.md) runs once
over the result. Bias-add — the eternal shape puzzle of positional APIs —
falls out with no ceremony (machine-verified, docs/REFERENCE.md §7.7,
context setup elided):

```zig
test "pointwise broadcasts by tag" {
    var x = try fucina.Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var bias = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();

    var y = try x.add(&ctx, &bias); // .d aligns; missing .batch broadcasts
    defer y.deinit();
    comptime std.debug.assert(@TypeOf(y).axis_tags.len == 2); // tags {.batch, .d}
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 14, 25, 36 }, try y.dataConst());
}
```

And because "compatible" means "union fits `max_rank`", *disjoint* tag sets
broadcast too: `rows.mul(&ctx, &cols)` for `rows: Tensor(.{.batch})` and
`cols: Tensor(.{.d})` yields result tags `{.batch, .d}` — "an outer product
without any reshape ceremony", machine-verified in docs/REFERENCE.md §7.7.

**The operand-order gotcha.** "Union in operand order" means the *left*
operand's tags come first, so operand order determines the physical layout
of the result: `{.d}` + `{.batch, .d}` produces tags `{.d, .batch}`, *not*
`{.batch, .d}` (docs/REFERENCE.md §7.4). `bias.add(ctx, &x)` and
`x.add(ctx, &bias)` compute the same numbers into two different layouts —
and two different *types*. Mathematically harmless, occasionally a
performance decision, always visible in the type. Say it once and remember
it: **result layout depends on operand order.**

Our mini `add` implements the same three-step rule over the mini `View`:
align both operands to the union tags, resolve each axis size (`if (da ==
db) da else if (da == 1) db else if (db == 1) da else return
error.ShapeMismatch` — names matched, sizes did not), then one flat loop
with broadcast axes clamped to coordinate 0. The full ~45-line
implementation is compile-checked, with tests pinning the bias-add, the
disjoint-tag union (an outer *sum*, since our mini op is add), and the
runtime `ShapeMismatch` — the same three behaviors the REFERENCE.md
snippets pin for the real library.

## 4.7 Contraction by name: `dot` and the compile error you came for

Matrix multiplication is where positional shape discipline actually hurts,
so it is where the tag algebra pays out hardest. Fucina's `dot` takes one
**contract tag** and derives everything else: the operands' tags partition
into three comptime classes (docs/REFERENCE.md §7.4):

- **batch tags** — present in both operands, not the contract tag;
- **left free tags** — left-only, non-contract;
- **right free tags** — right-only, non-contract;

and the result is `batch ++ left free ++ right free`. That one sentence
covers vector·vector (result `{}` — a scalar), matrix·vector,
matrix·matrix, and batched matmul, none of them special cases at the API.
The gatekeeper is `dotResultLen` (src/tags.zig:386–392):

```zig
pub fn dotResultLen(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) usize {
    _ = tagIndexOrCompileError(left_tags, contract_tag);
    _ = tagIndexOrCompileError(right_tags, contract_tag);
    const len = dotBatchLen(left_tags, right_tags, contract_tag) + dotLeftFreeLen(left_tags, right_tags, contract_tag) + dotRightFreeLen(left_tags, right_tags, contract_tag);
    if (len > tensor_mod.max_rank) @compileError("too many tensor tags");
    return len;
}
```

The contract tag must exist in **both** operands before this function will
even report a length. And because the facade `dot` calls it in the *return
type position* (src/ag/tensor.zig:3318):

```zig
pub fn dot(self: *const Self, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag) !Tensor(dotResultTags(tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag))
```

a misaligned contraction fails while the compiler is still trying to work
out what type the call would return — before any function body exists.

### Watching it fail

Time to break it for real. The following was compiled against the tree with
Zig 0.16.0 (via a scratch `build_options` stub — scalar backend, no BLAS —
since we drive `zig test` by hand outside `zig build`):

```zig
const fucina = @import("fucina");

pub fn bad(
    ctx: *fucina.ExecContext,
    a: *const fucina.Tensor(.{ .m, .k }),
    b: *const fucina.Tensor(.{ .n, .j }),   // no .k axis anywhere
) !void {
    var c = try a.dot(ctx, b, .k);
    defer c.deinit();
}
```

The compiler's verdict, verbatim (absolute path prefixes trimmed):

```
src/tags.zig:187:5: error: tensor tag not found
    @compileError("tensor tag not found");
    ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src/tags.zig:388:31: note: called at comptime here
    _ = tagIndexOrCompileError(right_tags, contract_tag);
        ~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~
src/tags.zig:368:122: note: called at comptime here
pub fn dotResultTags(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) [dotResultLen(left_tags, right_tags, contract_tag)]Tag {
                                                                                                             ~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
src/ag/tensor.zig:3318:123: note: generic function instantiated here
        pub fn dot(self: *const Self, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag) !Tensor(dotResultTags(tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag)) {
                                                                                                             ~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
repro_facade_error.zig:8:22: note: generic function instantiated here
    var c = try a.dot(ctx, b, .k);
                ~~~~~^~~~~~~~~~~~
```

Read the note chain bottom-up — it is the teaching gold. Your call site
instantiated `dot`; `dot`'s **return type** called `dotResultTags`; *its*
return type called `dotResultLen`; which demanded `.k` exist in the right
operand's tags; it doesn't; compilation over. The message is domain
vocabulary, the trace names your line, and no wrong-shaped tensor was ever
describable, let alone constructed.

> **ML note** — Compare the standard failure mode: the framework happily
> contracts positions, the shapes work out by coincidence, and you discover
> the transposed weight three layers deep, from a loss curve. Here the
> *meaning* error is caught in the only place it can be caught with
> certainty — before the program exists. The *size* error (§4.4) still
> waits at runtime; that division of labour is the whole design.

### The mini dot

Our course-code version implements the same idea minus batch axes (it
`@compileError`s on shared non-contract tags — promoting them to batch axes
is this chapter's hardest exercise):

```zig
/// Mini dot result tags: left free ++ right free. The contract tag must be
/// present in BOTH operands — otherwise this @compileErrors, and since it
/// runs in `dot`'s return type, the bad call site does too.
pub fn dotResultTags(
    comptime left: anytype,
    comptime right: anytype,
    comptime contract: Tag,
) [left.len + right.len - 2]Tag {
    _ = tagIndexOrCompileError(left, contract);
    _ = tagIndexOrCompileError(right, contract);
    inline for (left) |tag| {
        if (comptime !tagEqual(tag, contract) and tagIndex(right, tag) != null)
            @compileError("mini dot: shared non-contract tags (batch axes) not supported");
    }
    return removeTag(left, contract) ++ removeTag(right, contract);
}

/// Naive tag-directed contraction: every axis decision is a NAME lookup.
/// O(result * k) triple loop — chapter 6 is about doing this fast.
pub fn dot(
    a: anytype,
    b: anytype,
    comptime contract: Tag,
    out: []f32,
) error{ShapeMismatch}!View(dotResultTags(@TypeOf(a).axis_tags, @TypeOf(b).axis_tags, contract)) {
    const A = @TypeOf(a);
    const B = @TypeOf(b);
    const out_tags = comptime dotResultTags(A.axis_tags, B.axis_tags, contract);

    const ka = comptime tagIndexOrCompileError(A.axis_tags, contract);
    const kb = comptime tagIndexOrCompileError(B.axis_tags, contract);
    if (a.shape[ka] != b.shape[kb]) return error.ShapeMismatch; // the runtime half
    const k_dim = a.shape[ka];

    // ... result shape: each out tag reads its size from whichever operand
    //     owns it (comptime tagIndex decides; runtime array fills) ...

    for (0..total) |flat| {
        // ... decompose `flat` into result coords (row-major odometer) ...
        var acc: f32 = 0;
        for (0..k_dim) |k| {
            var ai: [A.rank]usize = undefined;
            inline for (A.axis_tags, 0..) |tag, i| {
                ai[i] = if (comptime tagEqual(tag, contract))
                    k
                else
                    coords[comptime tagIndexOrCompileError(out_tags, tag)];
            }
            // ... bi is built the same way from B.axis_tags ...
            acc += a.at(ai) * b.at(bi);
        }
        out[flat] = acc;
    }
    return View(out_tags).fromSlice(out[0..total], shape);
}
```

(Three mechanical stretches are elided; the full ~70-line function is in
the compile-checked scratch file, tests passing.) Read the inner `inline for`
carefully — it is the chapter's thesis in four lines. To build an operand's
index vector, each axis asks *at compile time*: "am I the contract axis?"
If yes, take the runtime loop variable `k`; if no, look up my tag's position
in the output *at compile time* and take that runtime coordinate. The
generated machine code contains only array indexing with constant offsets;
every name has dissolved.

> **Zig note** — Note the pattern `if (comptime tagEqual(tag, contract)) k
> else coords[comptime tagIndexOrCompileError(out_tags, tag)]`. When the
> condition is comptime-known, the untaken branch is not analyzed — so the
> `tagIndexOrCompileError(out_tags, contract)` that would fail (the contract
> tag is *not* in the output) never fires for the contract-axis iteration.
> Comptime-known branches are how you write per-axis special cases without
> runtime cost — the same trick `normalizeTags` used with `@hasField`.

Breaking the mini version reproduces the same shape of diagnostic. Feeding
it `View(.{ .m, .k })` against `View(.{ .n, .j })` stops compilation with
`axes.zig: error: tensor tag not found`, and the note chain walks the same
path: return type → `dotResultTags` → the failed lookup (captured from
`zig test`, Zig 0.16.0). Seventy lines of course code buy the same
guarantee as the real library.

### Semantics by name, kernels by layout

Because the contraction is *named*, "the operands' physical axis order never
changes the mathematical result — it only selects the kernel"
(docs/REFERENCE.md §7.9). The canonical example, machine-verified
(docs/REFERENCE.md §7.9): weights stored `{.n, .k}` — output-major, the way
linear layers usually keep them — contract against `{.m, .k}` with no
transpose in sight:

```zig
test "dot contracts by tag regardless of physical axis order" {
    var a = try fucina.Tensor(.{ .m, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .n, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 7, 9, 11, 8, 10, 12 });
    defer b.deinit();

    var c = try a.dot(&ctx, &b, .k); // lowers to the trans-B matmul kernel
    defer c.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, try c.dataConst());
}
```

(Context setup elided; full block in REFERENCE.md §7.9.) In a positional
API, that call requires you to transpose `b` — and getting it silently
wrong is the classic shape bug. Here `.k` finds itself on both sides; the
layout difference merely routes to `matmulTransB` instead of `matmul2D`.
Shared non-contract tags ride along as batch axes, and contracting the only
tag of two vectors yields `Tensor(.{})` — both pinned by machine-verified
snippets in REFERENCE.md §7.9.

Two boundaries to keep straight, both documented:

- **Dot batching is exact-match, not broadcast.** Batch sizes must be equal
  on both sides — they "do **not** broadcast (unlike pointwise)"
  (docs/REFERENCE.md §7.9). The escape hatch when you *do* want a stride-0
  broadcast batch is the facade `matmul` with an explicit result spelled
  out — `matmul(ctx, other, comptime kind, comptime out_tags) !Tensor(out_tags)`
  (src/ag/tensor.zig:456): you name the output tags, the library obliges.
- **Sizes still fail at runtime.** Our mini `dot` returns
  `error.ShapeMismatch` when `.k` is 3 on the left and 2 on the right, and
  so does the real one (§4.4). Names were never going to catch that.

## 4.8 Split, merge, flatten: structural algebra on names

The multi-head attention reshape is the scariest line in most transformer
implementations: `x.view(B, T, H, D // H).transpose(1, 2)` — four
positional numbers and an axis swap, unchecked. With named axes it is one
call that *factors an axis into named factors* (machine-verified,
docs/REFERENCE.md §7.8, context setup elided):

```zig
test "split and merge rename factor axes as views" {
    var x = try fucina.Tensor(.{ .batch, .d_model }).fromSlice(&ctx, .{ 2, 6 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    var heads = try x.split(&ctx, .d_model, .{ .head, .head_dim }, .{ 2, 3 });
    defer heads.deinit(); // tags {.batch, .head, .head_dim}, shape {2,2,3}
    const hs = heads.shape();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3 }, &hs);

    var flat = try heads.merge(&ctx, .features, .{ .head, .head_dim });
    defer flat.deinit(); // tags {.batch, .features}, shape {2,6}
    try std.testing.expectEqualSlices(f32, try x.dataConst(), try flat.dataConst());
}
```

`.d_model` becomes `.head, .head_dim`; downstream attention code contracts
over `.head_dim` and batches over `.head` *by name*
([Chapter 12](12-a-transformer-from-scratch.md) runs this for real inside
qwen3). The tag halves are pure tuple rewrites in `src/tags.zig` —
`splitTags` (:539) and `mergeTags` (:563), with compile errors for a split
tag colliding with an existing name or merge tags that are not "contiguous
and in tensor order". The size halves are where split and merge stop being
symmetric, and the asymmetry is a design decision worth internalizing
(docs/REFERENCE.md §7.8):

- **`split` is zero-copy on *any* source layout.** Factor strides derive
  from the split axis's own stride, so even a strided view splits fine. The
  runtime checks are only that factors are non-zero and multiply exactly to
  the source axis size (`TensorError.InvalidShape` otherwise).
- **`merge` is zero-copy *only* when the merged axes are
  stride-compatible** — laid out as an unsplit axis, i.e. for each adjacent
  pair `stride(i) == shape(i+1) × stride(i+1)`. A transposed or gapped
  layout returns `TensorError.UnsupportedView` — **no silent
  materialization**; you make the copy explicit with the facade
  `materialize` first. Costly copies stay visible in the program text.

The failing case is pinned too (machine-verified, docs/REFERENCE.md §7.8,
context setup elided):

```zig
test "merge rejects stride-incompatible layouts" {
    var x = try fucina.Tensor(.{ .a, .b }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var t = try x.permuteTo(&ctx, .{ .b, .a }); // zero-copy transposed view
    defer t.deinit();

    // Tag-contiguous, but the transposed strides cannot collapse into one axis.
    try std.testing.expectError(error.UnsupportedView, t.merge(&ctx, .m, .{ .b, .a }));
}
```

This is the chapter's two-worlds table (§4.4) in one op: *tag* contiguity
(the merge tags appear adjacently, in order) is checked at compile time;
*stride* contiguity (the layout actually collapses) is the runtime half of
the same rule.

Rounding out the structural set (facade signatures in
src/ag/tensor.zig:1095–1148, :1784–1798): `flatten(ctx, out_tag)` returns
`Tensor(.{out_tag})` — zero-copy when contiguous, materializing otherwise —
and `sumMany(ctx, reduce_tags)` reduces away several named axes at once,
returning `Tensor(removeTags(tags, reduce_tags))`; `sumAll` collapses
everything to `Tensor(.{})`. One implementation detail of `sumMany` is a
gem for the Zig audience: reduction strips one axis at a time, and removing
axis 1 would shift axis 3's index, so the axis indices are pre-sorted
*descending* — by a bubble sort that runs inside the compiler
(src/tags.zig:17–35):

```zig
pub fn reduceAxesDescending(comptime tags: anytype, comptime reduce_tags: anytype) [reduce_tags.len]usize {
    var axes: [reduce_tags.len]usize = undefined;
    inline for (reduce_tags, 0..) |tag, i| {
        axes[i] = tagIndexOrCompileError(tags, tag);
    }

    comptime var i: usize = 0;
    inline while (i < axes.len) : (i += 1) {
        comptime var j: usize = i + 1;
        inline while (j < axes.len) : (j += 1) {
            if (axes[j] > axes[i]) {
                const tmp = axes[i];
                axes[i] = axes[j];
                axes[j] = tmp;
            }
        }
    }
    return axes;
}
```

`comptime var`, `inline while`, swaps — an honest O(n²) sort, free at
runtime, n ≤ 8. When people say "Zig's comptime is just Zig", this is what
they mean: the sort you would write anyway, executed at another time.

## 4.9 Einsum: everything was a special case

One level up, the chapter's machinery unifies. The general two-operand
contraction — einsum — asks: given left tags, right tags, and *output*
tags, what role does each axis play? Fucina answers with a 2×2 truth table
(src/tags.zig:470–477):

```zig
pub const EinsumPart = enum { batch, contract, left_free, right_free, left_summed, right_summed };

pub fn einsumClassOfLeft(comptime right_tags: anytype, comptime out_tags: anytype, comptime tag: Tag) EinsumPart {
    const shared = tagIndex(right_tags, tag) != null;
    const kept = tagIndex(out_tags, tag) != null;
    if (shared) return if (kept) .batch else .contract;
    return if (kept) .left_free else .left_summed;
}
```

Shared and kept: batch. Shared and dropped: contracted. Private and kept:
free. Private and dropped: summed away first. Every matmul, batched matmul,
inner product, outer product, and marginalization is a row of this table —
and `dot`, our hero of §4.7, turns out to be a one-liner
(src/tagged.zig:105–118, parameters elided):

```zig
pub fn taggedDot(...) !RawTensor {
    return taggedEinsum(left_tags, left, ctx, right_tags, right, comptime dotResultTags(left_tags, right_tags, contract_tag));
}
```

`dot` is einsum with the canonical output order (batch ++ left free ++
right free) as the equation. "One einsum lowering serves every contraction
(`einsum`, `dot`, and their backward records)" (docs/ARCHITECTURE.md,
Current Strengths) — even the gradients: the backward branches of
`EinsumBackward` "are einsums themselves — the gradient of a contraction is
a contraction" (docs/REFERENCE.md §7.9). Hold that thought for
[Chapter 7](07-autograd.md).

On the facade, `einsum` takes the output tags directly —
`x.einsum(ctx, &y, .{ .batch, .m, .n })` — and its comptime guards speak in
prose: a quantized RHS gets `@compileError("einsum does not take a
quantized RHS; use dot, whose packed kernels require the [free, contract]
weight layout")` (src/ag/tensor.zig:3361–3362) — the message carries both
the *why* and the fix.

### Comptime plans, runtime choices

The einsum lowering is also where the chapter's central split — semantics at
comptime, layout strategy at runtime — appears in its most advanced form.
The tag algebra fixed *which* axes are M, N, K at compile time. Which
*kernel* runs (plain GEMM, transposed-A, transposed-B) depends on where the
data physically lies — a runtime fact. So the lowering builds zero-copy
aligned views of each operand against comptime tuple-concatenated targets —
`const x_plain_target = comptime batch_ord ++ m_ord ++ k_ord;`
(src/tagged.zig:624–627) — and *probes*, at runtime, which orientation is
already contiguous (src/tagged.zig:629–652). A transposed-kernel flag costs
nothing; a materialization costs a copy pass; the probe pays the flag
whenever it can, and when *both* operands want the transposed kernel, the
larger keeps it and the smaller materializes once. The batch group then
collapses by zero-copy reshape into a single bmm axis, and one kernel call
runs. How those kernels work — and why GEMM dominates everything — is
[Chapter 5](05-the-operation-library.md) and
[Chapter 6](06-going-fast-on-cpus.md)'s story.

One more documented cost (doc comment, src/tagged.zig:132–135): an
`out_tags` order that interleaves the batch/free groups pays one extra
output materialization — prefer group-nested orders. And even the
*rejected* fourth kernel is a documented decision: the double-transposed
GEMM (`trans_ab`) was measured and declined — "zero reachable sites" in
production code, and "the equation-level role-swap workaround is faster
than the trans_ab ceiling itself" (docs/BENCHMARK.md, a dated,
machine-specific record). Even the thing this library *didn't* build has a
benchmark trail; [Chapter 16](16-the-craft.md) returns to that culture.

## 4.10 The tags compile away entirely

The claim from §4.1 deserves a proper closing argument. **There is
deliberately no tagged tensor type at runtime.** The module doc of
`src/tagged.zig` (lines 10–13) states the decision:

> There is intentionally no tagged tensor *type* here: tags are comptime-only
> data (`tags.zig`), so the single runtime tensor currency stays the raw
> tensor (`tensor.zig`), which every heterogeneous container (autograd tape,
> ExecContext ops, weight unions) is built on.

A *runtime* tag representation would tax every container that holds tensors
of assorted shapes — the autograd tape, a weight table, a KV cache — with
either dynamic tag storage (per-instance memory, per-op comparisons) or
generic containers fragmented per tag set. Instead, everything below the
facade traffics in one concrete raw type, and the facade "re-attaches
result tags at comptime after each op" (docs/REFERENCE.md §7): the typed
and untyped worlds meet at function boundaries, where comptime information
is free.

You have already seen the facade's fields (§4.3) and measured our mini
`View` with `@sizeOf`. There is no tag array to store because there is no
moment at runtime when anyone needs to *ask* a tensor for its tags — every
question that mentions a name was answered during compilation, and what
remains are integer axis indices baked into the instructions.

Two guard rails complete the picture. First, the raw tensor is sealed off:
"a comptime guard in `src/fucina.zig` makes `fucina.RawTensor` a compile
error; in-tree code that genuinely needs it names
`fucina.internal.RawTensor`" (docs/REFERENCE.md §3) — you cannot
accidentally drop below the named-axis discipline. Second, the facade is
cheap enough to be the *only* public currency: "The no-grad facade has
negligible forward overhead, so model and example code carries
`fucina.Tensor(spec)` end-to-end" (docs/REFERENCE.md §3 — a qualitative
statement; the docs publish no numeric overhead figure, so none is quoted
here).

Which brings back the Fortran analogy one last time. `real A(n,m)` put the
rank in the program text and the compiler held you to it; dynamic
frameworks traded that away for flexibility and got shape bugs as the
exchange rate. Comptime lets this library refuse the trade: the discipline
of the old world, the ergonomics of the new, and a binary that contains
neither a tag nor a check for one.

> **Zig note** — Count what this chapter used: enum literals, `@TypeOf`,
> `@tagName`, `anytype`, `inline for`/`inline while`, `comptime var`,
> `@compileError`, `@hasField`, `@typeInfo`, functions returning types,
> dependent array lengths, comptime-known branching, tuple concatenation
> with `++`. Not one is a "metaprogramming feature" bolted to the side;
> each is ordinary Zig evaluated at compile time. (One pragmatic footnote:
> heavy comptime work can exhaust the interpreter's default budget, which
> is why the module occasionally calls `@setEvalBranchQuota` —
> e.g. src/tags.zig:282.)

## What you now know

- An axis name in Fucina is a **tag** — an enum literal, a comptime-only
  value with spelling equality (`Tag = @TypeOf(.tag)`, src/tags.zig:4).
- `Tensor(spec)` is a **function returning a type**; specs that normalize to
  the same (dtype, tag list) are the *same* type across spellings, while
  reordered tags are a *different* type (docs/REFERENCE.md §3.1,
  machine-verified).
- The type carries **names, not extents**: missing/duplicate tags, rank
  overflow (> 8), and wrong contraction axes fail at **compile time**;
  same-name axes with different sizes fail at **runtime** with
  `error.ShapeMismatch`.
- Result tags are computed by pure comptime functions *in return type
  position*: pointwise = union in operand order (so **operand order decides
  result layout**), dot = batch ++ left free ++ right free, contract tag
  required in both operands — the compile error fires before any body runs.
- Alignment, permutation, and broadcasting are **zero-copy views**: a
  missing tag becomes a size-1, stride-0 phantom axis — one element read
  many times instead of copied.
- `split` is zero-copy on any layout; `merge` refuses stride-incompatible
  layouts with `error.UnsupportedView` rather than silently materializing.
- Dot batching is exact-match (no broadcast); `matmul` with explicit
  `out_tags` is the escape hatch. Physical axis order never changes results
  — it only selects kernels, probed at runtime.
- `dot` is a one-line special case of einsum, whose axis roles come from a
  2×2 comptime table (shared × kept); one contraction engine serves forward
  and backward alike.
- The tags **compile away entirely**: the runtime struct is a raw tensor, a
  grad pointer, and a bool; there is deliberately no tagged tensor type.

## Explore the source

- `src/tags.zig` — the entire comptime algebra in ~600 lines; readable top
  to bottom, and the best `comptime` tutorial in the tree.
- `src/tags_tests.zig` / `src/tagged_tests.zig` — the algebra's rules
  pinned as tests, comptime and runtime halves respectively.
- `src/tagged.zig` — the runtime half: `alignTensorToOf` (the workhorse
  view), `pointwise`, `taggedDot`/`taggedEinsum` with the orientation probe.
  Remember: internal — user code goes through `Tensor` methods.
- `src/ag/tensor.zig:185–211` — the facade constructor: normalize, validate,
  `return struct`; then skim any view method's return type.
- `docs/REFERENCE.md` §3.1–3.2 and §7 — the semantics contract, with
  machine-verified snippets for every behavior this chapter claimed.
- `examples/spirals/main.zig` — the tags at work in a full training program
  you can run.
- `src/lora.zig` — `Adapter(in_tag, out_tag)`: tags as generic parameters of
  a whole *module*, not just a tensor (docs/ARCHITECTURE.md, layer map).

## Exercises

1. **Warm-up.** Add `hasTag(comptime tag: Tag) bool` and a comptime
   `axis(comptime tag: Tag) usize` to the mini `View`, mirroring the facade
   (src/ag/tensor.zig:976–984). Assert their results in a `comptime` block.
2. **Operand order.** Using the real library, compute `bias.add(&ctx, &x)`
   and `x.add(&ctx, &bias)` for `bias: Tensor(.{.d})`, `x: Tensor(.{ .batch,
   .d })`. Predict both result *types* before compiling (§4.6's gotcha),
   then verify with `comptime std.debug.assert(@TypeOf(...) == ...)`.
3. **One dispatcher, six ops.** Generalize the mini `add` into
   `pointwise(comptime op: enum { add, sub, mul, div }, a, b, out)`,
   selecting the scalar operation with a comptime `switch` — the shape of
   `src/tagged.zig`'s `pointwise` (:55–81). The broadcast machinery should
   not change at all.
4. **Batch axes (hard).** Remove the mini dot's `@compileError` on shared
   non-contract tags: implement `dotBatchTags` (shared, non-contract, in
   left order), make the result `batch ++ left free ++ right free`, and
   extend the odometer loop so batch coordinates index *both* operands.
   Check against the machine-verified batch example in REFERENCE.md §7.9.
5. **Merge, both halves (hard).** Give the mini `View` a `merge(out_tag,
   merge_tags)`: comptime-check the merge tags are adjacent and in order in
   `axis_tags`, then runtime-check `stride(i) == shape(i+1) × stride(i+1)`
   per adjacent pair, returning `error.UnsupportedView` on failure — and
   confirm a transposed view refuses, like docs/REFERENCE.md §7.8's snippet.

---

[Previous: Tensors from scratch](03-tensors-from-scratch.md) ·
[Next: The operation library](05-the-operation-library.md)
