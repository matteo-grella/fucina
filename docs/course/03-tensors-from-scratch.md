# Chapter 03 — Tensors from scratch

*Part II — The tensor core*

[Chapter 02](02-just-enough-ml.md) ended with a claim: the tensor is *the* data
structure of machine learning. Every model you will meet in this course — the
spiral classifier, the guitar amp, the transformer — is a pipeline of functions
from tensors to tensors. So before we can build any of them, we have to build
the tensor itself.

This chapter does that twice. First we build a miniature tensor from nothing —
a dtype tag, a flat buffer, shape and strides, zero-copy views, a refcount, a
buffer pool — in small compilable steps, so that every design decision arrives
as the answer to a problem you have just watched appear. Then we meet the real
thing: `src/dtype.zig`, `src/storage.zig`, and `src/tensor.zig`, the three
files at the bottom of Fucina's stack, and `docs/MEMORY-MODEL.md`, the document
that records *why* the memory model looks the way it does.

One honest framing before we start: the raw tensor layer we study here is
**deliberately not public API** — exporting it at the root is a compile error
(§3.13), and the reference manual warns "Expect this surface to change without
compatibility notice" (docs/REFERENCE.md §8). Everything in this chapter is
"how it is built", not "what you should call"; the supported surface is the
tagged facade of [Chapter 04](04-axes-with-names.md). We study the internals
anyway, because building them is the best way to understand any tensor
library you will ever use.

## 3.1 A tensor is flat memory plus an interpretation

Forget, for a moment, everything you know about multidimensional arrays. Here
is the entire idea:

```
memory:          [ 1  2  3  4  5  6 ]        one flat buffer, six f32s

interpretation:  shape   = {2, 3}            "read it as 2 rows of 3"
                 strides = {3, 1}            "a row is 3 elements apart,
                                              a column is 1 element apart"

                 ┌ 1  2  3 ┐
                 └ 4  5  6 ┘
```

A tensor is a flat, one-dimensional block of memory plus a few integers that
say how to *read* it as an n-dimensional array. The element at index
`[i, j, k, ...]` lives at:

```
address = offset + i·strides[0] + j·strides[1] + k·strides[2] + ...
```

That formula is the whole theory. NumPy works this way. PyTorch works this
way. Fucina works this way. Once you have internalized it, three magical-
looking operations become obvious bookkeeping:

- **Transpose** — swap two entries of `shape` and the same two entries of
  `strides`. No data moves.
- **Broadcast** — give an axis stride 0, so every index along it reads the
  same memory. No data is duplicated.
- **Slicing/narrowing** — bump `offset` and shrink `shape`. No data is copied.

The rest of this chapter is the engineering required to make that formula
safe, fast, and shareable in a language with no garbage collector.

> **ML note** — Why not nested arrays (`[][]f32`)? Locality: one contiguous
> block is what the prefetcher and [Chapter 06](06-going-fast-on-cpus.md)'s
> SIMD kernels want. Views: nested arrays cannot express "the same data,
> transposed" without copying. Uniformity: one allocation per tensor means
> the allocator, the refcount, and later the buffer pool manage one object.

## 3.2 Step 1: a name for the bytes — the dtype tag

Our flat buffer holds *bytes*. Before anything else we need a vocabulary for
what those bytes mean: 32-bit floats? 16-bit floats? 8-bit integers? A packed
block of 4-bit quantized weights? In Fucina that vocabulary is a single enum
called `DType`, and the crucial design decision is that **a dtype is a format
tag, not a Zig type**. It is a plain enum value; comptime functions map it to
real Zig types when code needs them.

Here is the miniature (course code — all "course code" in this chapter is
compile-checked with `zig test` on Zig 0.16.0; it is *not* repo code):

```zig
// Course code — a miniature of the idea behind src/dtype.zig.
const std = @import("std");

/// A dtype is a *format tag*, not a Zig type: it names how bytes encode numbers.
const DType = enum { f32, f64, i32, q4 }; // q4 stands in for a block-quantized format

/// A comptime function that maps the tag to a real Zig type.
fn Scalar(comptime dtype: DType) type {
    return switch (dtype) {
        .f32 => f32,
        .f64 => f64,
        .i32 => i32,
        .q4 => @compileError("block-quantized dtypes have no per-element scalar type"),
    };
}

test "Scalar returns a type at compile time" {
    comptime std.debug.assert(Scalar(.f32) == f32);
    comptime std.debug.assert(Scalar(.i32) == i32);

    // Scalar(dtype) is usable anywhere a type is expected:
    const buf: [4]Scalar(.i32) = .{ 1, 2, 3, 4 };
    try std.testing.expectEqual(@as(i32, 4), buf[3]);

    // Scalar(.q4) — uncommenting this is a compile error, by design.
}
```

Two Zig ideas carry this snippet, and they carry the whole library.

> **Zig note** — *Functions that return types are Zig's generics.* `Scalar` is
> an ordinary function; it just happens to take a `comptime` parameter and
> return `type`. There is no template syntax, no separate generics language —
> `switch` on an enum, returning types, evaluated at compile time. And
> `@compileError` in a switch arm is *API policy*: asking for the per-element
> type of a packed quantized format is not a runtime error to handle, it is a
> program that should never compile. Zig only analyzes the arms you actually
> use, so the error fires exactly when someone writes the bad call.

> **ML note** — Why does a tensor library need so many dtypes? Because on
> CPUs, **memory bandwidth is the budget**. An f32 weight costs 4 bytes to
> fetch; bf16 halves that; a 4-bit block format like Q4_K costs about 0.56
> bytes per weight (144 bytes per 256 elements — see the table in
> docs/REFERENCE.md §8.1). For large models, moving bytes — not multiplying
> numbers — is what you wait for. [Chapter 11](11-model-files-and-quantization.md)
> is devoted to those packed formats; this chapter only needs to *count* them
> correctly.

## 3.3 Step 2: a flat buffer on the heap

Now the storage. We want a heap object that owns a typed slice and remembers
which allocator created it, so that tearing it down needs no external context:

```zig
/// The storage: a heap header plus a flat typed slice. (Course code.)
const Buffer = struct {
    allocator: Allocator,
    data: []f32,
    refs: std.atomic.Value(u32), // ignore this field until §3.6

    fn create(allocator: Allocator, len: usize) !*Buffer {
        const self = try allocator.create(Buffer);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .data = try allocator.alloc(f32, len),
            .refs = std.atomic.Value(u32).init(1),
        };
        return self;
    }
};
```

Note the shape of the thing: the buffer is a *header* (`allocator.create`
gives it a stable heap address) pointing at a separately allocated data
slice. That stable address matters in a moment, when several tensors need to
point at the same header. And yes, I have smuggled in a `refs` field —
pretend you did not see it; we earn it in §3.6.

> **Zig note** — Two idioms from [Chapter 01](01-just-enough-zig.md) are
> everywhere in Fucina. First, *the allocator travels with the object*:
> `Buffer` stores the allocator that made it and uses it at teardown, so a
> buffer can be freed from anywhere without anyone remembering which
> allocator to use. Second, `errdefer allocator.destroy(self)`: if the
> *second* allocation (`alloc(f32, len)`) fails, the first one is rolled back
> automatically, and only on the error path. Constructor choreography like
> this — allocate, `errdefer` the rollback, allocate the next thing — is the
> manual-memory equivalent of exception safety, visible in `create` at
> src/storage.zig:25-35.

## 3.4 Step 3: shape and strides — the interpretation

The tensor itself is astonishingly small: a pointer to a buffer plus the
interpretation metadata, stored *inline* in fixed-size arrays.

```zig
/// The tensor: a buffer pointer plus inline interpretation metadata. (Course code.)
const max_rank = 4;

const Tensor = struct {
    buffer: *Buffer,
    rank: usize,
    shape: [max_rank]usize,
    strides: [max_rank]usize,
    offset: usize,

    /// The one formula everything rests on.
    fn addressOf(self: *const Tensor, index: []const usize) usize {
        var addr = self.offset;
        for (index, 0..) |ix, i| addr += ix * self.strides[i];
        return addr;
    }

    fn at(self: *const Tensor, index: []const usize) f32 {
        return self.buffer.data[self.addressOf(index)];
    }
};
```

Why fixed arrays instead of slices? Because then a tensor is a plain *value*:
creating a view (or passing a tensor around) allocates nothing. Fucina makes
the same call with two `[8]usize` arrays — a raw tensor is exactly four
fields, `buffer`, `shape`, `strides`, `offset` (src/tensor.zig:95-98), and
the docs call it out: "no allocation per view" (docs/REFERENCE.md §8.5).
Rank above 8 is simply not representable, and nobody has missed it.

Fresh tensors get **row-major** ("C order") strides: the last axis is
contiguous, and each earlier axis strides by the product of everything after
it. For shape `{2, 3}` that is strides `{3, 1}`; for `{4, 2, 3}` it is
`{6, 3, 1}`. The code that computes this is nine lines, and it is worth
staring at until it is obvious — from `src/tensor.zig:639-647`:

```zig
fn writeContiguousStrides(out: []usize, shape: []const usize) void {
    var stride: usize = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        out[i] = stride;
        stride *= shape[i];
    }
}
```

Walk the shape backwards, carrying a running product. That's it.

**Contiguity** is the exact inverse question: are this tensor's strides
*precisely* the row-major strides of its shape? From `src/tensor.zig:289-298`:

```zig
pub fn isContiguous(self: *const Self) bool {
    var expected: usize = 1;
    var i = self.shape.len;
    while (i > 0) {
        i -= 1;
        if (self.strides.at(i) != expected) return false;
        expected *= self.shape.at(i);
    }
    return true;
}
```

Contiguity is *the license to touch memory as a flat slice*. If (and only if)
a tensor is contiguous, its logical elements and its physical elements are the
same sequence, and handing out `buffer.data[offset .. offset + len]` is
meaningful. Every fast path in the library — `memcpy`, SIMD kernels, GEMM —
starts by establishing contiguity. Non-contiguous tensors must either be
walked stride-by-stride or *materialized* (copied into fresh contiguous
storage) first.

Here is the compile-checked test from our miniature:

```zig
test "strides turn flat memory into an n-d array" {
    const alloc = std.testing.allocator;
    var x = try Tensor.zeros(alloc, &.{ 2, 3 });
    defer x.deinit();
    for (x.buffer.data, 0..) |*v, i| v.* = @floatFromInt(i); // 0,1,2,3,4,5

    // shape {2,3} => strides {3,1}: element [1][2] lives at 1*3 + 2*1 = 5.
    try std.testing.expectEqual(@as(usize, 3), x.strides[0]);
    try std.testing.expectEqual(@as(f32, 5), x.at(&.{ 1, 2 }));
    try std.testing.expect(x.isContiguous());
}
```

> **ML note** — "Row-major" has a practical consequence you will use
> constantly: *the last axis is the cheap one*. Summing along the last axis
> reads memory sequentially; summing along the first hops by large strides.
> When [Chapter 05](05-the-operation-library.md) introduces reductions and
> [Chapter 12](12-a-transformer-from-scratch.md) lays out attention matrices,
> "which axis is innermost" is a performance decision, not a notational one.

## 3.5 Step 4: views — transpose, broadcast, reshape without copying

Now the payoff of the strides design. A **view** is a new tensor value — new
shape, new strides, possibly new offset — pointing at the *same* buffer.

**Transpose** is a metadata swap (course code):

```zig
/// Transpose: swap the metadata, never touch the data. (Course code.)
fn transpose(self: *const Tensor) Tensor {
    std.debug.assert(self.rank == 2);
    self.buffer.retain(); // §3.6 explains this line
    var out = self.*;
    out.shape[0] = self.shape[1];
    out.shape[1] = self.shape[0];
    out.strides[0] = self.strides[1];
    out.strides[1] = self.strides[0];
    return out;
}
```

A `{2,3}` tensor with strides `{3,1}` becomes a `{3,2}` tensor with strides
`{1,3}`. Same six floats. Reading `t[j][i]` computes `j·1 + i·3` — exactly
the address `x[i][j]` computed before. Note what `isContiguous` says about
the result: strides `{1,3}` are *not* the row-major strides of shape
`{3,2}` (those would be `{2,1}`), so the transposed view is non-contiguous —
correct, and the reason `data()` will refuse to hand out a flat slice for it.

**Broadcast** is the stride-0 trick, and it is worth seeing Fucina's real
implementation because NumPy's celebrated broadcasting rules turn out to be
fifteen lines. From `src/tensor.zig:464-483` (inside `broadcastFromRankToRank`;
`rank_diff` is how many new leading axes the target adds):

```zig
inline for (0..target_rank) |target_i| {
    if (target_i < rank_diff) {
        target_strides[target_i] = 0;
    } else {
        const source_i = target_i - rank_diff;
        const source_dim = source.shape[source_i];
        const target_dim = target_shape[target_i];
        if (source_dim == target_dim) {
            target_strides[target_i] = source.strides[source_i];
        } else if (source_dim == 1) {
            target_strides[target_i] = 0;
        } else {
            return TensorError.ShapeMismatch;
        }
    }
}

self.buffer.retain();
errdefer self.buffer.release();
return initFromBufferWithStrides(tensor_dtype, self.buffer, target_shape[0..], target_strides[0..], self.offset);
```

Read the three cases aloud: a brand-new leading axis gets stride 0 (every
index reads the same place); a matching axis keeps its stride; a size-1 axis
gets stride 0 (the single element repeats); anything else is a shape error.
Right-aligned, exactly like NumPy. **No data is touched.** A `{3}` row
broadcast to `{4, 3}` is still three floats in memory, read twelve times.

> **Zig note** — `inline for` unrolls the loop at compile time — legal here
> because `target_rank` is a comptime parameter; each iteration is stamped
> out as straight-line code with `target_i` a constant. §3.10 shows how a
> *runtime* rank becomes a comptime one, so tricks like this apply everywhere.

**Reshape** is the subtle one. Viewing `{2, 3}` as `{6}` or `{3, 2}` without
copying is only possible when the logical order of elements equals the
physical order — that is, when the tensor is contiguous. So `reshape` in
Fucina *requires* contiguity and returns `error.UnsupportedView` otherwise
(src/tensor.zig:224-236); a transposed view must be materialized before it
can be reshaped. A reshape that succeeds is pure relabeling: same buffer,
same offset, new shape, fresh row-major strides.

And now the problem we created. Run the transpose test:

```zig
test "transpose is a stride swap, not a copy" {
    const alloc = std.testing.allocator;
    var x = try Tensor.zeros(alloc, &.{ 2, 3 });
    defer x.deinit();
    for (x.buffer.data, 0..) |*v, i| v.* = @floatFromInt(i);

    var t = x.transpose();
    defer t.deinit();
    try std.testing.expect(t.buffer == x.buffer); // same storage, zero bytes moved
    try std.testing.expect(!t.isContiguous());
    try std.testing.expectEqual(x.at(&.{ 1, 2 }), t.at(&.{ 2, 1 }));

    t.set(&.{ 0, 1 }, 42); // writes are visible through every alias
    try std.testing.expectEqual(@as(f32, 42), x.at(&.{ 1, 0 }));
}
```

Two independent tensor values, one buffer. Both have a `deinit`. Who frees
the memory? If `x.deinit()` frees the buffer, `t` dangles. If neither does,
we leak. If both do, we double-free. In a garbage-collected language this
question doesn't exist; in Zig it *is* the design problem.

## 3.6 Step 5: refcounting — exactly one destructor

The answer is the oldest one in systems programming: count the owners. The
buffer carries an atomic reference count; every legitimately owned tensor
value holds exactly one reference; every view constructor takes one more
(`retain`); every `deinit` gives one back (`release`); whoever returns the
*last* reference destroys the storage — whether that is the parent or the
view, in either order.

The subtle part is "whoever returns the last reference", because releases can
race across threads. Here is Fucina's actual `release`, from
`src/storage.zig:117-129`, and it repays close reading:

```zig
pub fn release(self: *Self) void {
    const old = self.refs.fetchSub(1, .acq_rel);
    std.debug.assert(old > 0);
    if (old == 1) {
        self.discardPending();
        self.waitUnused();
        if (self.release_fn) |release_fn| {
            release_fn(self.release_ctx.?, self);
        } else {
            self.destroy();
        }
    }
}
```

`fetchSub` atomically decrements **and returns the previous value**. If two
threads release simultaneously, the hardware serializes them: exactly one of
them gets back `old == 1`, and exactly that one runs the destructor. No lock,
no double-free, no leak. The `debug.assert(old > 0)` turns an over-release —
releasing a buffer nobody owns — into a loud failure in safety builds instead
of silent corruption. (The `discardPending`/`waitUnused` calls and the
`release_fn` hook are real-library concerns we take up in §3.7 and §3.9.)

> **Zig note** — Why do the atomic orderings differ? `retain` uses
> `.monotonic` (src/storage.zig:107-109): taking a reference needs no
> synchronization beyond the counter itself, because whoever gave you the
> tensor already synchronized with you. `release` uses `.acq_rel`: the
> *release* half makes all your writes to the data visible before the count
> drops, and the *acquire* half makes sure the final decrementer sees
> everyone else's writes before destroying. Retain-monotonic /
> release-acq_rel is the standard refcount recipe, in eleven readable lines.

Our miniature's `release` is the same shape minus the hook, and the payoff
test — the one that would be a double-free without refcounting — is the
broadcast view outliving its parent (course code, compile-checked):

```zig
test "broadcast is stride zero; a view outlives its parent" {
    const alloc = std.testing.allocator;
    var row = try Tensor.zeros(alloc, &.{3});
    for (row.buffer.data, 0..) |*v, i| v.* = @floatFromInt(i + 1); // 1,2,3

    var grid = try row.broadcastTo(&.{ 4, 3 }); // {3} -> {4,3}, no copy
    defer grid.deinit();
    try std.testing.expectEqual(@as(usize, 0), grid.strides[0]);

    row.deinit(); // the buffer survives: grid still holds a reference
    try std.testing.expectEqual(@as(f32, 3), grid.at(&.{ 2, 2 }));
    try std.testing.expectEqual(@as(f32, 1), grid.at(&.{ 3, 0 }));
}
```

`std.testing.allocator` fails the test if anything leaks — so this test
proves both directions: the buffer survived `row.deinit()`, and it was freed
exactly once by the end.

One rule completes the model, and it is a rule of *discipline*, not
mechanism: **copying the tensor struct does not retain.** `var y = x;` gives
you a second struct pointing at the same buffer with no extra reference —
deinit both and you double-free. Only the sanctioned constructors and view
methods create owned values. In exchange, `deinit` poisons the struct
(`self.* = undefined`, src/tensor.zig:176-179), so a dead tensor holds
recognizable garbage (Debug builds fill undefined memory with a poison
pattern) and using — or re-deiniting — one typically crashes loudly rather
than quietly reading freed memory: a debugging aid, not a guaranteed check.

## 3.7 Step 6: a buffer pool — deinit as a recycling driver

Our tensor now works. But watch what an eager runtime does with it. Every
operation in Fucina allocates a fresh output tensor (there is no graph
compiler planning memory — [Chapter 05](05-the-operation-library.md) makes
this precise), so a forward pass is a storm of same-sized allocate/free
pairs: allocate a `{batch, hidden}` activation, use it, free it, allocate
another `{batch, hidden}` activation... The general-purpose allocator is
neither told nor able to exploit the pattern.

The fix is a **free list**: when a buffer's refcount hits zero, don't free it
— park it in a pool, and let the next same-sized request take it back. The
elegant part is *where* this plugs in. Look again at `release` above: at
refcount zero it calls `release_fn(release_ctx, self)` **instead of**
destroying, if a hook is set. The destructor is pluggable. A pool doesn't
need to wrap the tensor, intercept `deinit`, or exist in the tensor's
vocabulary at all — it just installs itself as the buffer's release hook at
creation time.

Here is the miniature pool (course code, compile-checked):

```zig
const Pool = struct {
    allocator: Allocator,
    free_list: std.ArrayList(*Buffer),

    fn acquire(self: *Pool, len: usize) !*Buffer {
        for (self.free_list.items, 0..) |buffer, i| {
            if (buffer.data.len >= len) { // first fit
                _ = self.free_list.orderedRemove(i);
                buffer.refs.store(1, .release); // hand it out as freshly owned
                return buffer;
            }
        }
        const buffer = try Buffer.create(self.allocator, len);
        buffer.release_ctx = self;
        buffer.release_fn = reclaim; // wire recycling into the refcount
        return buffer;
    }

    /// Runs at refcount zero instead of free(): back to the free list.
    fn reclaim(ctx: *anyopaque, buffer: *Buffer) void {
        const self: *Pool = @ptrCast(@alignCast(ctx));
        self.free_list.append(self.allocator, buffer) catch buffer.destroy();
    }
};

test "release recycles through the pool: the same address comes back" {
    const alloc = std.testing.allocator;
    var pool = Pool.init(alloc);
    defer pool.deinit();

    const first = try pool.acquire(256);
    const first_ptr = first.data.ptr;
    first.release(); // refcount 0 -> reclaim -> free list, NOT free()

    const second = try pool.acquire(256); // same size: same buffer back
    defer second.release();
    try std.testing.expectEqual(first_ptr, second.data.ptr);
}
```

The test's assertion is the important one: **pointer equality**. Release a
buffer, ask for the same size, get the *same address* back. That is not just
saved allocator work — it is cache warmth. The buffer you just wrote your
activation into is still in L1/L2 when the next op writes its output to the
very same lines. We will see Fucina assert exactly this property in §3.12.

> **Zig note** — `release_ctx: ?*anyopaque` plus a function pointer is Zig's
> manual version of a closure: `*anyopaque` is a type-erased pointer (C's
> `void*`, but cast back with `@ptrCast(@alignCast(...))` under alignment
> rules). The buffer knows nothing about pools; it just calls whatever hook
> it was given, with whatever context it was given. The same two fields let
> a buffer wrap GPU-resident bytes or a user's mmap — one mechanism, many
> owners (§3.9).

The from-scratch build is complete: tag, buffer, strides, views, refcount,
pool — six steps, each forced by the previous one. Now, the production versions.

## 3.8 The real dtype system (`src/dtype.zig`)

Fucina's `DType` has 37 tags (src/dtype.zig:3-41, abridged):

```zig
pub const DType = enum {
    bool,
    u8, u16, i8, i16, i32, i64,
    f16, bf16, f32, f64,
    q1_0, q4_0, q4_1, q5_0, q5_1, q8_0, q8_1,
    q2_k, q3_k, q4_k, q5_k, q6_k, q8_k,
    iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs,
    tq1_0, tq2_0, mxfp4, nvfp4,
};
```

Every tag is one of two *kinds* (`DTypeKind`, src/dtype.zig:49): **scalar**
(one storage element per logical element — the first eleven) or
**block-quantized** (one packed struct per block of 32–256 logical elements —
the other 26, all GGML-compatible wire formats). Three comptime type
functions map tags to Zig types, and their domains encode policy:

- `Scalar(dtype)` (src/dtype.zig:250) — the per-logical-element type.
  Compile error for block dtypes, exactly like our miniature's `.q4` arm.
- `Storage(dtype)` (src/dtype.zig:293) — the per-*storage*-element type:
  `Scalar(dtype)` for scalars, the block struct for block formats. Buffers
  are sized in `Storage(dtype)` units — which is why our from-scratch layer
  can carry quantized data it cannot decode: it only has to count blocks.
- `Accumulator(dtype)` (src/dtype.zig:325) — the type reductions accumulate
  in: `f32` for `.f16`/`.bf16`/`.f32`, `f64` for `.f64`, `i64`/`u64` for the
  integers.

Two entries deserve a pause.

**`Scalar(.bf16) == u16`.** bfloat16 is stored and passed as *raw bits*, not
a float type. Why that is sane becomes clear from the conversion functions,
which are a small gem — `src/dtype.zig:750-763`, complete:

```zig
pub fn bf16ToF32(bits: u16) f32 {
    const widened: u32 = @as(u32, bits) << 16;
    return @bitCast(widened);
}

pub fn f32ToBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if ((bits & 0x7fff_ffff) > 0x7f80_0000) {
        return @truncate((bits >> 16) | 64);
    }
    const lsb = (bits >> 16) & 1;
    const rounded = bits + 0x7fff + lsb;
    return @truncate(rounded >> 16);
}
```

**bf16 is literally the top 16 bits of an f32** — same sign bit, same 8
exponent bits, mantissa truncated from 23 bits to 7. Widening is one shift.
That is bf16's entire reason to exist: it keeps f32's *range* (no overflow
surprises during training) at half the memory, and conversion costs almost
nothing. Narrowing is three lines of integer arithmetic implementing
round-to-nearest-even (`+ 0x7fff + lsb` — add just under half, plus one more
when the kept LSB is odd, so ties go to even), and the first branch quiets
NaNs (`| 64` sets a mantissa bit so a NaN payload never truncates to
infinity — ggml-compatible, pinned by `src/dtype_tests.zig`). Even the
constant `one(.bf16) == 0x3f80` (src/dtype.zig:611-617) makes sense now:
it is the top half of f32's `0x3f80_0000`, which is 1.0. Compare f16, which
spends its 16 bits on more mantissa and less exponent: more precision,
much smaller range.

**The block structs are wire formats.** Each block dtype maps to an
`extern struct` — C layout, no padding games — and a `comptime` block pins
every one of their sizes at build time (struct at src/dtype.zig:74-77,
comments added; sizes pinned at :221-248, abridged):

```zig
pub const BlockQ8_0 = extern struct {
    d: u16,                       // one shared f16 scale...
    qs: [q8_0_block_size]i8,      // ...and 32 small integers
};

comptime {
    std.debug.assert(@sizeOf(BlockQ8_0) == 34);
    std.debug.assert(@sizeOf(BlockQ4_K) == 144);
    // ... all 26 formats ...
}
```

If a struct ever drifts from the byte-exact GGML layout, the library stops
compiling. That is a *test that cannot be skipped* — [Chapter 16](16-the-craft.md)
collects this verification style. The one-sentence intuition for what a block
*means* — one shared scale plus small integers, value ≈ scale × int — is all
we need until [Chapter 11](11-model-files-and-quantization.md) decodes them
properly. To the tensor layer of this chapter, a `BlockQ4_K` is an opaque
144-byte payload.

Finally, dtype policy. Two comptime functions, `computeDType` and
`outputDType` (src/dtype.zig:564, :585), encode one policy table for the
whole op library of [Chapter 05](05-the-operation-library.md). The rules
that matter now:

- **f32 is the compute currency.** bf16 *always* computes through f32 (it is
  raw bits; there is nothing to compute with until you widen).
- **Accumulate wider than you store.** Reductions over `.f16`/`.bf16` return
  `.f32` — summing ten thousand halves into a half would throw away the
  accumulator's precision at the last step. Integer reductions return `.i64`.
  This is visible in public result *types*, and a machine-verified snippet in
  docs/REFERENCE.md §8.3 pins it.
- **Nothing promotes silently.** Mixed-dtype math is a compile error on the
  facade; casts are explicit ops. Where a cast must exist, its edge cases are
  *defined*: `castScalar` documents that float→int truncates toward zero and
  saturates at the target bounds with NaN → 0 — "defined everywhere — torch's
  CPU float-to-int overflow is unspecified" (src/dtype.zig:714-719).
- **Only float dtypes can carry gradients** (`supportsGrad`,
  src/dtype.zig:430) — and in practice only `.f32` does
  (docs/REFERENCE.md §8.2). Gradients are [Chapter 07](07-autograd.md)'s story.

> **ML note** — "Accumulate wider than you store" is likely your first
> numerical-precision lesson, and it generalizes: the *storage* dtype is a
> bandwidth decision, the *accumulator* dtype is a correctness decision, and
> good libraries separate them. You will meet the same split inside the GEMM
> kernels of [Chapter 06](06-going-fast-on-cpus.md) and the mixed-precision
> training of [Chapter 08](08-training.md).

## 3.9 The real storage (`src/storage.zig`)

The production buffer is our miniature plus the hook fields plus three GPU
fields, generic over dtype — `src/storage.zig:8-23`:

```zig
pub fn BufferOf(comptime buffer_dtype: DType) type {
    const Elem = dtype_mod.Storage(buffer_dtype);

    return struct {
        allocator: Allocator,
        data: []Elem,
        refs: std.atomic.Value(u32),
        release_ctx: ?*anyopaque = null,
        release_fn: ?*const fn (*anyopaque, *Self) void = null,
        pending_work: std.atomic.Value(?*accelerator.Work) = .init(null),
        pending_use: std.atomic.Value(?*accelerator.Work) = .init(null),
        accelerator_resource: std.atomic.Value(?*accelerator.Resource) = .init(null),

        const Self = @This();
        pub const dtype = buffer_dtype;
        pub const Element = Elem;
```

`BufferOf` is a comptime function returning a struct type — one source, a
family of concrete buffer types, with `pub const Buffer = BufferOf(.f32);`
(src/storage.zig:234) as the workhorse alias. Note `pub const dtype` *inside*
the struct: types can carry constants, so any code holding a buffer type can
ask it what format it stores at compile time.

The three `pending_*`/`accelerator_resource` fields are **out of scope for
this chapter**: they are fences and cache metadata for asynchronous GPU
offload (Metal/CUDA), which enters the story as a backend seam much later.
All you need here is that they exist on the buffer and that `release` and
`destroy` drain them before storage is recycled — host-side code like ours
never sees them non-null on a CPU build.

What *is* this chapter's business is the constructor family, because it
enumerates every ownership pattern the library needs (all return `refs == 1`;
the table is docs/REFERENCE.md §8.4):

| Constructor | Data ownership | At `refs == 0` |
|---|---|---|
| `create(allocator, len)` | owned, uninitialized | free data + header |
| `createWithRelease(...)` | owned | run the hook |
| `fromSlice(allocator, values)` | owned copy | free data + header |
| `fromBorrowedSlice(allocator, values)` | **aliases caller memory** | free header only |
| `fromBorrowedSliceWithRelease*(...)` | aliases | hook, with **full cleanup duty** |

The borrowed variants are how zero-copy loading works: a buffer can wrap
bytes it does not own — a caller's array, an mmap'd weight file, GPU-resident
memory — and the refcount machinery works identically; only what happens at
zero differs. Two contracts keep this sound:

- **Borrowed bytes must outlive the tensor, unmoved.** The buffer aliases
  them; nobody copies anything.
- **A release hook takes full cleanup duty**: dispose of the external bytes
  *and* free the header via `destroyHeader()` — and call `destroyHeader()`
  *before* freeing the bytes, because provider teardown may still need the
  live address (docs/REFERENCE.md §8.4 shows a complete worked example that
  wraps an mmap and munmaps it from the hook, exactly once, at the last
  release). One caution from the same section: Fucina's own GGUF loader does
  *not* use hooks for mmap'd weights — that lifetime is holder-managed
  ([Chapter 11](11-model-files-and-quantization.md)); the hook is the tool
  for *user* code wanting refcount-driven cleanup.

Thread-safety has a crisp boundary worth memorizing: `retain`/`release` are
atomic and may race freely from any thread; the **data slice is not
synchronized** — concurrent reads are fine, writers need external
coordination (the runtime's parallel kernels only ever partition disjoint
ranges — [Chapter 06](06-going-fast-on-cpus.md)). And `isUnique()`
(src/storage.zig:113-115) is explicitly a *snapshot*, not a lock — the
source comment says "Use for ownership-transfer APIs when the caller already
has exclusive access". We meet its canonical consumer in §3.10.

## 3.10 The real tensor (`src/tensor.zig`)

The value type, `src/tensor.zig:94-101` and `:176-179`:

```zig
return struct {
    buffer: *Buffer,
    shape: Shape,
    strides: Shape,
    offset: usize = 0,

    const Self = @This();
    pub const dtype = tensor_dtype;
    // ...
    pub fn deinit(self: *Self) void {
        self.buffer.release();
        self.* = undefined;
    }
```

`Shape` is a `u8` rank plus an inline `[max_rank]usize` with `max_rank = 8`
(src/tensor.zig:9,21-23) — our miniature's design, hardened. Everything is
measured in *storage elements* (for block dtypes, strides count blocks). Two
deliberate restrictions:

- **No rank 0.** `Shape.init` rejects empty shapes and zero-sized dims
  (src/tensor.zig:25-34) — a zero-size tensor is unrepresentable by
  construction. A "scalar" is a rank-1 `{1}` tensor; the facade's scalar-tag
  type wraps exactly that.
- **Block dtypes admit only whole-tensor views.** `reshape` and the
  `viewWithStrides*` family accept only the identity view for block formats
  (src/tensor.zig:201-205) — blocks are indivisible, so you cannot slice
  through the middle of one.

**Checked views.** Our miniature's `transpose` trusted its caller. The real
general-view constructor proves safety instead — `src/tensor.zig:200-222`,
scalar path:

```zig
pub fn viewWithStridesOffset(self: *const Self, shape: []const usize, strides: []const usize, offset_delta: usize) !Self {
    // ... block-dtype identity-view arm elided ...
    _ = try elementCount(shape);
    if (strides.len != shape.len) return TensorError.InvalidShape;

    const view_offset = try std.math.add(usize, self.offset, offset_delta);
    var max_index = view_offset;
    for (shape, strides) |dim, stride| {
        const span = try std.math.mul(usize, dim - 1, stride);
        max_index = try std.math.add(usize, max_index, span);
    }
    if (max_index >= self.buffer.data.len) return TensorError.InvalidDataLength;

    self.buffer.retain();
    errdefer self.buffer.release();
    return initFromBufferWithStrides(tensor_dtype, self.buffer, shape, strides, view_offset);
}
```

The loop computes the **maximal reachable index**: the address formula is
monotone in every index, so its maximum over the whole view is at
`index[i] = dim[i]−1` for all i, i.e. `offset + Σ (dim−1)·stride`. If that
lands inside the buffer, *every* address the view can generate does — one
check makes arbitrary strided views memory-safe. `std.math.add`/`mul` return
errors on overflow, so even a malicious shape cannot wrap the arithmetic
into a false pass. Then the retain-then-`errdefer`-release choreography you
now recognize from every view.

One thing you will *not* find here: a `narrow` (slice-along-an-axis) method.
Narrowing is `viewWithStridesOffset` with a bumped offset and a smaller
shape, and the library builds it at the exec/facade layers rather than
duplicating it on the raw type (docs/REFERENCE.md §8.5.3). The primitive is
the general view; everything else is derived.

**Data access, honest about contiguity.** Four accessors split one decision
two ways (src/tensor.zig:307-329): `data()`/`dataConst()` return the flat
slice and **panic** on non-contiguous tensors — they are for hot paths that
have already established contiguity, and the panic message tells you the fix
("materialize or use dataChecked"). `dataChecked()`/`dataConstChecked()`
return `error.UnsupportedView` instead — recoverable, for code that cannot
know. Corollary worth writing down: a broadcast `{1}` view with stride 0 is
*not* contiguous by the definition in §3.4, so even `item()` — which reads
through `dataConst` — panics on it. When you need the contents of an
arbitrary view, `copyTo` walks the strides, and `clone(allocator)` is
materialization: fresh contiguous buffer, `copyTo` into it, done
(src/tensor.zig:181-188).

For the curious, the strided walk itself, `copyRangeTo`
(src/tensor.zig:352-421), is a small masterclass: it finds the maximal
row-major-contiguous *suffix* of axes and moves that as whole `@memcpy` runs,
advances the outer coordinates like an odometer with incremental stride
arithmetic — one coordinate decode per call, never a division per element —
and its header documents that disjoint ranges may be copied concurrently,
which is exactly how the runtime parallelizes big materializations.

**The ownership optimization.** `canTakeInPlace` (src/tensor.zig:300-305)
returns true for a full-buffer, contiguous, uniquely-referenced tensor —
the conditions under which a consuming op may *steal* the operand's buffer
and write its result in place instead of allocating. Read the comment above
it carefully: the refcount check proves no other tensor aliases the buffer
*right now*, and the answer is trustworthy **only while the caller has
exclusive access to the tensor handle**. It is a snapshot, not
synchronization — nothing stops another thread that also holds the handle
from retaining a moment later; the contract forbids that situation rather
than detecting it. Neither `isUnique` nor `canTakeInPlace` is a thread-safety
mechanism.

**Runtime rank meets comptime rank.** Rank lives in a `u8` at runtime, but
§3.5's `inline for` needed rank at comptime. The bridge is an eight-way
switch — `src/tensor.zig:503-515`:

```zig
fn dispatchRank(comptime tensor_dtype: DType, comptime F: anytype, rank: usize, args: anytype) !TensorOf(tensor_dtype) {
    return switch (rank) {
        1 => @call(.auto, F, .{1} ++ args),
        2 => @call(.auto, F, .{2} ++ args),
        // ... 3 through 8 ...
        else => TensorError.InvalidShape,
    };
}
```

Each arm passes a *comptime-known* rank, so the compiler stamps out one
specialized copy of `F` per rank actually used. `rankView(comptime rank)`
(src/tensor.zig:250-265) is the same idea as a value: it copies shape and
strides into `[rank]usize` arrays so kernels can unroll — note that it
*borrows* the tensor without retaining, so the ranked view must not outlive
it. This pair is how [Chapter 06](06-going-fast-on-cpus.md)'s kernels get
comptime shapes out of runtime tensors.

The reference manual's machine-verified "raw view tour" ties the whole layer
together (docs/REFERENCE.md §8.5.3, abridged):

```zig
var x = try RawTensor.fromSlice(alloc, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
defer x.deinit();

// Transposed view: same buffer, swapped strides, not contiguous.
var t = try x.viewWithStrides(&.{ 3, 2 }, &.{ 1, 3 });
defer t.deinit();
std.debug.assert(t.buffer == x.buffer);
std.debug.assert(!t.isContiguous());

// t.data() would panic here; the checked accessor reports an error.
try std.testing.expectError(error.UnsupportedView, t.dataChecked());

// clone materializes any view into fresh contiguous storage.
var m = try t.clone(alloc);
defer m.deinit();
std.debug.assert(m.isContiguous() and m.canTakeInPlace());
```

## 3.11 Ownership end to end: who frees what

Climb one level. Applications do not call `Buffer.create`; they call *ops* on
an `ExecContext` (the runtime object [Chapter 05](05-the-operation-library.md)
dissects). What does ownership look like up there? One sentence, quoted from
docs/REFERENCE.md §6.2:

> "every tensor an op returns is owned by the caller and must be
> deinitialized exactly once — unless an exec scope is open, in which case op
> results are scope-owned borrows and `deinit` on them is a safe no-op."

Unpack the first half. Fucina is eager: an op call runs the kernel and
returns an owned result immediately — no graph object, no deferred execution
(docs/REFERENCE.md §6). So ownership is a per-value question answered at the
call site, and the universal idiom is:

```zig
var y = try someOp(&ctx, ...);
defer y.deinit();
```

Constructors (`fromSlice`, `zeros`, `variable`, ...) follow the same rule,
with one refinement: a facade constructor that *consumes* a raw tensor does
so **on success only** — on error, ownership stays with the caller, so your
`errdefer` arms remain correct (docs/REFERENCE.md §3.3). The same
success-only convention governs `fromOwnedBuffer` at the raw layer
(src/tensor.zig:163-169). Failures never leak, and never double-free.

For values that *evolve* — a residual stream flowing through transformer
layers, an accumulator in a loop — there is `ctx.replace(old, new_value)`:
deinit the old, rebind the new, in one statement, with the old value left
intact if the new one's computation failed (docs/REFERENCE.md §6.2 has the
verified snippet). You will use it constantly from
[Chapter 12](12-a-transformer-from-scratch.md) on.

Now the second half of the sentence — the exec-scope escape hatch — is a
preview of [Chapter 07](07-autograd.md), but the memory-model half of it
belongs here. Training cannot deinit-ASAP: every intermediate between the
parameters and the loss must survive until `backward()` has consumed it. Exec
scopes resolve this without forking the code: open a scope, and op results
become *scope-owned borrows* whose `deinit` is a safe no-op; close the scope
after backward and everything is released, newest first. The verified
snippet from docs/REFERENCE.md §6.3:

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

The consequence is what the docs call **write-once forward code**: the same
defer-deinit forward function serves inference (no scope: deinits recycle
eagerly) and training (scope open: deinits are no-ops, the scope holds
everything for backward). What scopes *cost*, and why you should never hold
one on an inference path, is §3.12's punchline. How they interact with the
autograd graph is [Chapter 07](07-autograd.md)'s.

> **ML note** — The two lifetime regimes are worth naming, because they are
> inherent to the mathematics, not to this library. *Inference*: each
> intermediate is needed only by the next op — free it as soon as its
> consumer has run, and the working set stays O(1) per block (the docs put
> the per-block peak live set at ~6–12 tensors, MEMORY-MODEL.md §4).
> *Training*: the backward pass re-reads the forward's activations —
> backward functions hold operand *views*, so the refcounts you built in
> §3.6 pin those buffers automatically until backward is done. "Wasting"
> memory on a buffer that will be read again is not waste; it is the
> definition of training (MEMORY-MODEL.md §2).

## 3.12 The memory model: why a pool, not an arena

Every experienced systems programmer who reads §3.11 asks the same question
within a minute: *why all this per-tensor deinit ceremony?* Why not an
**arena** — bump-allocate every tensor of the forward pass from a region and
free it all in one reset at the end? No refcounts, no defers, no double-free
class of bugs. It is a genuinely good idea, the right call in many programs,
and Fucina evaluated it seriously enough to write a design document
adjudicating it: `docs/MEMORY-MODEL.md`. Its verdict is worth quoting exactly:

> "The `BufferPool` already *is* an arena (it amortizes allocation), but a
> strictly better one — it additionally recycles buffers *within* a single
> forward pass, bounds steady-state memory, and coexists with refcounted
> views and the autograd tape." (docs/MEMORY-MODEL.md)

First, meet the real pool, because it is our §3.7 miniature with three
policies added (`src/exec/buffer_pool.zig`; one instance per context). The
acquire path, `:82-108`, is first-fit over a free list kept sorted ascending
by size — and on a miss, note it releases the mutex *before* allocating
fresh. Size rounding (`:273-278`) collapses nearby sizes into shared buckets
so reuse actually hits:

```zig
fn allocationLen(len: usize) usize {
    if (len <= 1024) {
        return std.math.ceilPowerOfTwo(usize, len) catch len;
    }
    return std.mem.alignForward(usize, len, 1024);
}
```

And within a size class, reclaim inserts *before* equal-sized entries — the
comment in the source states the rationale better than any paraphrase
(`src/exec/buffer_pool.zig:185-188`):

```zig
// Insert BEFORE equal-size entries: within a size class the pool
// hands back the most-recently-released buffer first (LIFO) — its
// lines are the likeliest to still be cache-resident. Ordering is
// address-only; numerics are unaffected.
```

Retention is bounded: `max_cached_bytes` defaults to 1 GiB
(src/exec/buffer_pool.zig:59 — raised from an original 64 MB because, per
the in-source comment, a single fused activation at 4096-token prefill on a
0.6B model is ~100 MB). A second arm serves non-f32 dtypes from 64-byte-
aligned byte slabs so an f16 scratch released by one op can serve q8_k
scratch in the next; both arms share the one budget. And the pool doubles as
a leak detector: an atomic `outstanding` counter must be zero at
`BufferPool.deinit` (:70) — leak a tensor past context teardown and safety
builds abort instead of shrugging.

Here is the memory model made *observable* — the machine-verified test from
docs/REFERENCE.md §6.2, and the single most striking demo in this subsystem:

```zig
test "deinit recycles transient buffers through the pool" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try ctx.fromSlice(&.{3}, &.{ 1, 2, 3 });
    defer a.deinit();

    var first = try ctx.add(&a, &a);
    const first_ptr = first.dataConst().ptr;
    first.deinit(); // storage returns to the pool free list

    var second = try ctx.add(&a, &a); // same size: the pool hands back the same address
    defer second.deinit();
    try std.testing.expectEqual(first_ptr, second.dataConst().ptr);
}
```

Pointer equality across two op calls. `deinit` is not bookkeeping overhead —
it is the *recycling driver*: `tensor.deinit()` → `buffer.release()` →
refcount 0 → release hook → free list → next op gets warm cache lines. Every
`defer x.deinit()` you write in this library is feeding that loop.

Now the adjudication. MEMORY-MODEL.md §4 rejects the arena on four
substantive axes; compressed:

1. **Peak memory regresses from working-set to sum-of-all-intermediates.**
   The pool reclaims each transient the instant it dies; the per-block live
   set is ~6–12 tensors, and the residual stream is one carried value. An
   arena frees nothing until reset, so a forward balloons to roughly
   `n_layer ×` the activation footprint.
2. **It destroys cache locality.** Bump allocation returns a fresh address
   per op; the pool returns the *same* address for same-sized successive
   allocations — the asserted behavior above, in direct service of the
   library's match-llama.cpp-on-CPU goal.
3. **It cannot express refcounted views** — the decisive constraint. The
   document's key example: per-step attention reads the KV cache through a
   zero-copy narrow aliasing a *session-lifetime* buffer
   ([Chapter 12](12-a-transformer-from-scratch.md)) — a view that, as
   docs/REFERENCE.md §6.2 puts it, "has a per-object lifetime no region
   reset can express". A reset would either free live KV memory or have to
   carve it out entirely, at which point it is no longer one arena.
4. **It is incorrect for training by construction** — activations must
   outlive the forward (§3.11's second regime), so an intra-forward reset
   frees exactly the memory backward is about to read.

And the closing line: "The pool already delivers the *only* real arena
advantage — allocation amortization — and adds intra-pass reuse plus a
bounded cap the arena lacks" (MEMORY-MODEL.md §4).

The document records a second adjudication (dated 2026-06-10) that closes
the remaining loophole: could exec scopes — which *are* arena-like ownership
— become the inference memory manager? Measured answer: no. A held scope
turns the pool's O(1) working set into O(N) live intermediates at cold
addresses — **2 versus 32 distinct buffers on a 32-op 1 MiB chain**, a
measurement pinned by the "exec scope holds buffers until close" test in
`src/ag/tensor_tests.zig` (MEMORY-MODEL.md §5). Scopes are for when holding
everything *is the semantics* (training); deinit-ASAP with no scope stays
the inference discipline. The document is equally honest about the pattern's
one real cost — boilerplate, ~21 `defer .deinit()` lines across a
transformer's attention + FFN blocks (§5) — and why the tempting "frame
helper" fix stays unbuilt until that actually bites.

> **Zig note** — Keep MEMORY-MODEL.md's *form* in mind as much as its
> content: a rejected design gets a dated document with file:line references
> "so the rationale can be re-verified rather than trusted", measured
> numbers, and hard preconditions under which the verdict would change. That
> is what [Chapter 16](16-the-craft.md) means by engineering discipline —
> and it is why this course can quote its numbers instead of inventing any.

## 3.13 Drawing the line: internals, not API

One last design decision, and it is about *restraint*. Everything you built
in this chapter works; why isn't it the public interface? Fucina answers by
making the question unaskable — from `src/fucina.zig:33-42`:

```zig
comptime {
    // Anti-regression guard: re-exporting the raw tensor type at the
    // PUBLIC ROOT is a COMPILE ERROR. This fires on any build that analyzes the
    // module root (every test/example/tool), not just `zig build test`. ...
    if (@hasDecl(@This(), "RawTensor")) @compileError(
        "fucina.RawTensor must not be exported at the public root; raw tensors are internal. " ++
            "Use fucina.internal.RawTensor (in-tree raw naming) or bench_raw.RawTensor (microbench).",
    );
}
```

A `comptime` block as an architectural test: if anyone ever adds
`pub const RawTensor` to the root, nothing in the tree compiles. The
documented rationale is "API shape, not capability" — the tagged facade adds
negligible forward overhead (the docs' qualitative characterization; no
figure is claimed), so a public raw type would only split the ecosystem into
two tensor vocabularies (docs/REFERENCE.md §8.6). Code that genuinely needs
the raw layer — backend kernels, format/byte work, the LLM band needing
exact type identity — names it through the sanctioned escape hatch,
`fucina.internal.RawTensor` (src/fucina.zig:128-167). For *inspection*, the
facade already crosses the boundary safely: every public tensor exposes
`asRawTensor()`, a read-only view of the raw metadata you now know how to
read.

Which is the note to end on. The public type you will use from the next
chapter onwards — `fucina.Tensor(.{ .batch, .in })` — is a wrapper around
exactly the four fields you built here: a buffer pointer, two inline `[8]usize`
arrays, an offset. What the wrapper adds is not machinery but *meaning*:
names for the axes, checked at compile time. That is
[Chapter 04](04-axes-with-names.md).

## What you now know

- A tensor is **flat memory plus an interpretation**: `address = offset +
  Σ index·stride`. Transpose, broadcast, and narrowing are stride/offset
  bookkeeping — zero-copy views over shared storage.
- A **dtype is a format tag**, not a Zig type; comptime functions
  (`Scalar`/`Storage`/`Accumulator`, src/dtype.zig) map tags to types, with
  `@compileError` making invalid mappings unrepresentable. bf16 is f32's top
  16 bits stored as raw `u16`; blocks are opaque `extern struct` wire formats
  pinned by comptime size asserts.
- **Contiguity** — strides exactly row-major for the shape — is the license
  to touch memory as a flat slice; `data()` panics without it,
  `dataChecked()` errors, `clone` materializes.
- **Refcounting with `fetchSub`**: the decrementer that observes `old == 1`
  runs the destructor exactly once, even under races; view constructors
  retain, `deinit` releases and poisons; copying the struct does not retain.
  `isUnique`/`canTakeInPlace` are snapshots under exclusive handle access,
  never thread-safety mechanisms.
- **Release hooks** make the destructor pluggable: the same refcount
  machinery manages owned memory, borrowed slices, mmap wraps, and pooled
  buffers.
- **Ownership contract**: every op result is caller-owned, deinit exactly
  once — except under an exec scope, where results are scope-owned borrows
  and deinit is a safe no-op (the training story of
  [Chapter 07](07-autograd.md)).
- **The buffer pool beats an arena** on peak memory, cache locality (same
  address back — pointer-equality-tested), refcounted views/KV aliasing, and
  training lifetimes; docs/MEMORY-MODEL.md records the adjudication, with
  the 2-vs-32-buffers scope measurement as the cautionary tale.
- The raw layer is **unstable internals by design**, guarded by a comptime
  anti-export check; the supported surface is the tagged facade.

## Explore the source

- `src/dtype.zig` — the 37-tag enum, the type-mapping trio, the dtype policy
  tables, and the bf16 bridge; the whole file is under 800 lines and readable
  in one sitting.
- `src/storage.zig` — `BufferOf`, the constructor/ownership matrix, and
  `release` with the exactly-once hook dispatch.
- `src/tensor.zig` — the four-field tensor value, checked views with the
  max-reachable-index proof, the odometer `copyRangeTo`, and `dispatchRank`.
- `src/exec/buffer_pool.zig` — both pool arms, the LIFO-for-cache-warmth
  comment, size rounding, and the `outstanding` leak assert.
- `docs/MEMORY-MODEL.md` — the arena adjudication; read it end to end as a
  model of how to document a design decision.
- `docs/REFERENCE.md` §8 (raw layer) and §6.2–6.5 (ownership, scopes, pool) —
  the machine-verified snippets this chapter quoted.

## Exercises

1. **(Easy)** Extend the course dtype miniature (§3.2) with `.u8` and a
   `fn sizeOf(comptime dtype: DType) usize` that returns the per-element byte
   size via `@sizeOf(Scalar(dtype))`. Verify with a test that
   `sizeOf(.f64) == 8`. Then look up how the real library answers the same
   question for block formats (`blockByteSize`, src/dtype.zig:560).
2. **(Easy)** In the mini-tensor, write `fn narrowRows(self: *const Tensor,
   start: usize, count: usize) !Tensor` returning a view of rows
   `[start, start+count)` of a rank-2 tensor: bump `offset` by
   `start * strides[0]`, shrink `shape[0]` to `count`, retain the buffer.
   Add a bounds check in the spirit of the max-reachable-index proof of
   §3.10, and a test proving the view aliases (writes through it are visible
   in the parent).
3. **(Medium)** Add `allocationLen`-style size rounding (§3.12) to the
   mini-pool of §3.7 and write a test showing that requests of 100 and 120
   elements reuse the *same* recycled buffer, while 100 and 2000 do not.
   Then explain in one paragraph why rounding *up* wastes memory but
   improves reuse — and where the real pool caps the waste.
4. **(Medium)** Write `fn materialize(self: *const Tensor, allocator:
   Allocator) !Tensor` for the mini-tensor: allocate a fresh contiguous
   buffer and copy the view's logical elements into it with a nested stride
   walk. Test it on a transposed view and confirm the result `isContiguous()`.
   Then read the real `copyRangeTo` (src/tensor.zig:352-421) and identify
   (a) which axes it absorbs into a `@memcpy` run and (b) why it never
   divides per element.
5. **(Hard)** Read `reclaimTypedFor` in `src/exec/buffer_pool.zig:250-266`
   together with the two comptime asserts in `acquireTyped` (:120-122).
   Reconstruct the argument for why rounding the typed view's byte length
   back up with `alignForward(len * @sizeOf(Elem), slab_size_quantum)`
   recovers the slab's *exact* original capacity — and construct the
   counterexample showing the reasoning would break if an element type were
   allowed to be larger than `slab_size_quantum`.

---

[Previous: Just enough machine learning](02-just-enough-ml.md) ·
[Next: Axes with names: types that know their shape](04-axes-with-names.md)
