# 3. Tensors: types, construction, and data access

`fucina.Tensor` is the single public tensor type. It is a comptime type
constructor: `Tensor(spec)` returns a struct type whose axis names (tags),
rank, and dtype are fixed at compile time, while axis *sizes* are runtime
values. Every tensor operation goes through an explicit `*ExecContext`
(see [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)); there is no global state. The raw, tag-less tensor underneath the
facade is deliberately **not** exported at the public root — a comptime guard
in `src/fucina.zig` makes `fucina.RawTensor` a compile error; in-tree code
that genuinely needs it names `fucina.internal.RawTensor` ([§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)). The no-grad
facade has negligible forward overhead, so model and example code carries
`fucina.Tensor(spec)` end-to-end.

Sources for this section: `src/ag/tensor.zig` (the facade), `src/tags.zig`
(spec normalization), `src/tensor.zig` (raw storage semantics visible through
the facade), `src/exec.zig` (raw-tensor producers). The ownership discipline
is documented in depth in [MEMORY-MODEL.md](../MEMORY-MODEL.md).

Snippets in this section are runnable test blocks and assume:

```zig
const std = @import("std");
const fucina = @import("fucina");
```

## 3.1 The `Tensor(spec)` type constructor (`src/ag/tensor.zig`, `src/tags.zig`)

```zig
pub fn Tensor(comptime spec: anytype) type
```

`spec` takes one of five forms, all normalized by `src/tags.zig` before
the branch is instantiated:

| Spec form | Example | Meaning |
|---|---|---|
| Named tag tuple | `Tensor(.{ .batch, .in })` | rank = tuple length, dtype `.f32` |
| Numeric rank | `Tensor(2)` | rank-2 f32 with auto tags `._0, ._1` |
| dtype + tags | `Tensor(.{ .dtype = .f16, .tags = .{ .seq, .d } })` | typed, named axes |
| dtype + rank | `Tensor(.{ .dtype = .i64, .rank = 2 })` | typed, auto tags `._0, ._1` |
| Scalar | `Tensor(.{})` | zero tags; stored as raw rank-1 shape `{1}` |

Tags are anonymous enum literals (`Tag = @TypeOf(.tag)`); any identifier
works (`.batch`, `.qkv`, `._0`). Auto tags for numeric-rank specs are
`._0 … ._7`. Rules enforced at compile time:

- **Unique tags.** Duplicate tags in one spec are a compile error
  (`validateUniqueTags`).
- **Max rank 8** (`tensor.max_rank`); more tags are a compile error.
- **dtype defaults to `.f32`** when the spec carries no `.dtype` field.
- A dtype-struct spec must include `.tags` or `.rank`.

**Type identity.** Specs that normalize to the same (dtype, tag list)
produce the *same* type: `Tensor(2) == Tensor(.{ ._0, ._1 })` and
`Tensor(.{ .batch, .in }) == Tensor(.{ .dtype = .f32, .tags = .{ .batch, .in } })`.
Tensors are therefore freely interchangeable across function boundaries
regardless of which spelling declared them.

**Comptime vs runtime.** Rank, tags, and dtype are comptime; sizes are
runtime. Every branch exposes the comptime constants

```zig
pub const axis_tags: [tag_count]Tag; // normalized tag list
pub const tag_count: usize;          // logical rank (0 for scalars)
pub const tensor_rank: usize;        // raw storage rank; rawRank(0) == 1
pub const dtype: DType;
```

**Scalars.** `Tensor(.{})` has `tag_count == 0` but `tensor_rank == 1`:
scalars are stored as a rank-1, single-element tensor of shape `{1}`
(zero-size and zero-rank raw tensors are not representable — `Shape.init`
rejects any zero dimension with `error.InvalidShape`; a deliberate design
stance, rationale in ARCHITECTURE.md "Tensor And Storage Model"). Full
reductions such as `sumAll` return `Tensor(.{})`.

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

## 3.2 The four facade branches (`src/ag/tensor.zig`)

`Tensor(spec)` comptime-dispatches on the dtype into four struct families.
The method set of each branch is decided at compile time, so calling an
unsupported operation is a compile error, never a runtime failure
(pinned by the `@hasDecl` guard tests in `src/ag/tensor_tests/facade.zig`):

| Branch | dtypes | Capabilities |
|---|---|---|
| Float (differentiable) | `.f32` | Full surface: autograd (`variable`, `backward`, `grad`), all math/NN ops ([§4](04-tensor-operations.md)), load-time dense `packRhs`, all views and structural ops |
| Typed float | `.f16`, `.bf16`, `.f64` (`supportsForwardFloatMath`) | Forward math: the native typed set (`add/sub/mul/div/sum/mean/sumAll/dot/scale/divScalar`, `to`), the shared view set of every scalar-dtype branch ([§3.10](03-tensors-types-construction-and-data-access.md#310-facade-surface-index); the diagonal/band family, the constant fills, and the index-tensor gathers/scatters stay f32-only), and — **f16/bf16 only** — the ops with an f32 kernel, run through the `widened` dtype policy from the mixins shared with the f32 branch (unary family, gated, scalar ops, masks, comparisons, softmax family, scans, the remaining reductions, `pad`, the norms, `matmul`, `einsum`), plus load-time dense `packRhs`; [§4.19](04-tensor-operations.md#419-math-on-non-f32-tensors-srcagtensorzig)). **f16/bf16 only, autograd LEAVES**: `variable`/`variableFromSlice` with f32 gradients; differentiable `to` casts and mixed-RHS `dot`/`einsum` are the graph entries ([§5.1](05-automatic-differentiation.md#51-the-gradient-model-srcagtensorzig-srcagcorezig)) |
| Typed scalar constant | `.bool`, `.u8`, `.u16`, `.i8`, `.i16`, `.i32`, `.i64`, `.f8_e4m3`, `.f8_e5m2` | Constants only; construction, data access, the shared view/structural set ([§3.10](03-tensors-types-construction-and-data-access.md#310-facade-surface-index)), `to` (scalar casts, [§3.8](03-tensors-types-construction-and-data-access.md#38-casting-todtype-srcagtensorzig-srcexecconvertzig)), and integer forward math ([§4.19](04-tensor-operations.md#419-math-on-non-f32-tensors-srcagtensorzig)): wrapping `add`/`sub`/`mul`, `maximum`/`minimum`, explicit `divTrunc`/`divFloor` + `rem`/`mod`, bitwise `bitAnd`/`bitOr`/`bitXor`, i64-returning `sum`/`sumAll`, plus exact integer `compare` ([§4.6](04-tensor-operations.md#46-masks-comparisons-and-conditionals-srcagtensorzig)). `.bool` keeps only `to`, the counting `sum`/`sumAll`, and the mask combinators `logicalAnd`/`logicalOr`/`logicalXor`/`logicalNot`. The f8 storage floats (`.f8_e4m3`/`.f8_e5m2`, [§8.1](08-data-types-storage-and-the-raw-tensor-layer-internal.md#81-the-dtype-enum-srcdtypezig)) keep only construction, data access, structural ops, and `to` VALUE casts through the f32 bridge (no integer or float forward math) |
| Block-quantized constant | `q1_0/q2_0`, `q4_0 … q8_k`, `iq*`, `tq1_0/tq2_0`, `mxfp4`, `nvfp4` (`isBlockQuantized`) | Constants only; block construction, `to(.f32)` dequantize, `getRows`, row `concat`, `packRhs`/`packRhsAs` ([§10](10-quantization.md)) |

Notes that follow from the dtype layer (`src/dtype.zig`, detailed in [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)):

- `Scalar(.bf16)` is `u16` — bf16 tensors store and expose **raw bits**, not
  a native float type; `f16` uses Zig's `f16`. The f8 storage floats follow
  the same convention with `u8` bits (`dtype_mod.f32ToF8e4m3` /
  `f8e4m3ToF32` and the e5m2 pair are the value bridges, [§8.2](08-data-types-storage-and-the-raw-tensor-layer-internal.md#82-storage-mapping-and-dtype-predicates-srcdtypezig)).
- Block-quantized tensors have no per-element scalar; their element type is
  the block struct (`Storage(dtype)`, e.g. `fucina.quant.BlockQ8_0`), and shapes
  count *logical* elements while storage counts blocks.
- Output dtypes of typed-float math follow `dtype_mod.outputDType`:
  pointwise and `dot` keep the input dtype; reductions (`sum`, `mean`,
  `sumAll`) widen `f16`/`bf16` to `f32` (`f64` stays `f64`).

Only the f32 and typed-float branches have gradient machinery: they carry
`grad_state: ?*GradState` and a `scope_owned: bool` flag (see [§3.4](03-tensors-types-construction-and-data-access.md#34-deinit-lifetime-and-exec-scopes-srcagtensorzig-srctensorzig); on the
typed-float branch only f16/bf16 tensors — autograd leaves and
differentiable cast results, [§5.1](05-automatic-differentiation.md#51-the-gradient-model-srcagtensorzig-srcagcorezig) — ever populate it); the typed scalar and
block-quantized branches hold just the raw value and hard-code
`requiresGrad() == false`.

## 3.3 Construction and ownership (`src/ag/tensor.zig`, `src/exec.zig`)

All constructors are associated functions of the concrete tensor type and
take `ctx: *ExecContext` first (uniform signature even where the context is
unused, as in `constant`). Shape parameters are fixed-size arrays
`[tensor_rank]usize`, so passing the wrong-rank literal is a compile error.

**The core ownership contract.** `variable` and `constant` *consume* a raw
tensor produced by the context (`ctx.fromSlice(.f32, ...)` and friends, [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)) **on
success**; on error, ownership stays with the caller. Every returned facade
tensor is owned by the caller and must be released with `deinit()` (idiom:
`var x = try ...; defer x.deinit();`). Constructor failures never leak: the
facade holds `errdefer value.deinit()` internally around validation.

f32 branch:

```zig
pub fn variable(ctx: *ExecContext, value: RawTensor) !Self            // trainable leaf
pub fn variableFromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self
pub fn constant(ctx: *ExecContext, value: RawTensor) !Self            // no-grad wrap
pub fn fromTensor(ctx: *ExecContext, value: RawTensor) !Self          // alias of constant
pub fn fromSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self
pub fn fromBorrowedSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []f32) !Self
pub fn fromBorrowedConstSlice(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const f32) !Self
pub fn empty(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self  // uninitialized
pub fn zeros(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self
pub fn ones(ctx: *ExecContext, raw_shape: [tensor_rank]usize) !Self
pub fn full(ctx: *ExecContext, raw_shape: [tensor_rank]usize, fill_value: f32) !Self
pub fn scalar(ctx: *ExecContext, scalar_value: f32) !Self             // single-element

pub fn emptyLike(self: *const Self, ctx: *ExecContext) !Self          // torch *_like
pub fn zerosLike(self: *const Self, ctx: *ExecContext) !Self
pub fn onesLike(self: *const Self, ctx: *ExecContext) !Self
pub fn fullLike(self: *const Self, ctx: *ExecContext, fill_value: f32) !Self

pub fn arange(ctx: *ExecContext, start: f32, end: f32, step: f32) !Self       // single-tag types only
pub fn linspace(ctx: *ExecContext, start: f32, end: f32, steps: usize) !Self  // single-tag types only
pub fn oneHot(ctx: *ExecContext, indices: []const usize, depth: usize) !Self  // two-tag types only
pub fn eye(ctx: *ExecContext, n: usize) !Self                                 // two-tag types only

pub fn rand(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self       // uniform [0, 1)
pub fn uniform(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, lo: f32, hi: f32) !Self
pub fn randn(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self      // standard normal
pub fn normal(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, mean_value: f32, std_dev: f32) !Self
pub fn bernoulli(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64, p: f32) !Self
pub fn gumbel(ctx: *ExecContext, raw_shape: [tensor_rank]usize, seed: u64) !Self      // standard Gumbel(0, 1)
```

Semantics:

- `variable` allocates a leaf `GradState`; the tensor participates in
  autograd ([§5](05-automatic-differentiation.md)). `variableFromSlice` is `ctx.fromSlice` + `variable`.
- `fromSlice` **copies** `values` into context-owned storage.
- `fromBorrowedSlice` **borrows** caller-owned mutable storage zero-copy:
  the slice must stay alive and unmoved until the tensor's `deinit`;
  mutations of the backing slice are visible through the tensor. On a GPU
  build, mutate through the tensor's `data()` boundary (or synchronize
  externally) after submitting an op: direct writes through the external
  slice cannot be observed by the storage reader fence.
- `fromBorrowedConstSlice` borrows **read-only** storage (e.g. mmap'd GGUF
  weights) without a caller-side `@constCast`. The single internal
  `@constCast` is sound only under the contract that the data is never
  mutated through `.data()`; use `fromSlice` if a writable buffer is needed.
- `empty` returns uninitialized, buffer-pool-backed storage ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)); `zeros`,
  `ones`, `full`, `scalar` initialize it.
- The `*Like` forms are instance sugar over the same constructors
  (torch `zeros_like` and friends): same tags and dtype (both are part of
  the tensor type), shape taken from the receiver's logical shape (strided
  views included). Like every constructor the result is a fresh no-grad
  constant — the receiver's grad state does not carry over — and is never
  scope-owned. The typed scalar/float branches get `emptyLike`/`zerosLike`/
  `onesLike` (matching their static set, which has no `full`).
- `arange` is torch.arange with float semantics: element i is
  `start + i·step` (not accumulated), the end exclusive; `step == 0` or an
  empty range errors `InvalidShape` (zero-size tensors are not
  representable). `linspace` is torch.linspace: `steps` evenly spaced
  values, end INCLUSIVE and pinned exactly (`steps == 1` yields
  `{start}`; `steps == 0` is `InvalidShape`). `oneHot` builds the f32
  `[indices.len, depth]` one-hot matrix (torch F.one_hot with an explicit
  class count) from host-side indices — first tag rows, second tag
  classes; `indices[i] >= depth` is `IndexOutOfBounds`. `eye` is the
  `[n, n]` identity matrix (torch.eye) — first tag rows, second tag
  columns; `n == 0` is `InvalidShape`.
- The random constructors draw from the deterministic counter-based
  stream at `seed` ([§6.8](06-the-execution-runtime-execcontext-and-the-memory-model.md#68-determinism-and-the-rng-contract-srcrngzig), `fucina.rng`): element i is a pure function of
  `(seed, i)`, so a stored seed regenerates the exact tensor — the stream
  IS the generator abstraction; pass a fresh seed per draw (reusing one
  reuses the values). `rand`/`uniform` map one stream output per element
  onto `[lo, hi)` (`uniformFill`); `randn`/`normal` are Box-Muller
  (`gaussianFill`/`normalFill`); `bernoulli` is 1.0 iff the `[0, 1)` draw
  at `(seed, i)` falls below `p` (`p` outside `[0, 1]` is
  `InvalidShape`); `gumbel` is standard Gumbel(0, 1) noise
  `-ln(-ln(u))` over a strictly open uniform (`gumbelFill` — every draw
  finite), the gumbel-max / gumbel-softmax building block: add to logits
  and `argmax` for a categorical sample, or `softmax` at a temperature
  for its differentiable relaxation. All are no-grad constants, like
  every constructor.
- The typed **i64** branch adds two seed-stream constructors of its own
  (the repo-wide index dtype; [§3.10](03-tensors-types-construction-and-data-access.md#310-facade-surface-index)): `randint(ctx, raw_shape, seed, low, high)`
  — uniform integers over `[low, high)` via the widening multiply-shift
  map (`randintFill`; `low >= high` is `InvalidShape`) — and
  `randperm(ctx, n, seed)` — a rank-1 Fisher–Yates permutation of
  `{0, …, n-1}` (`randpermFill`; single-tag types only). The `.bool`
  branch adds the `bandMask` attention-mask constructor ([§4.6](04-tensor-operations.md#46-masks-comparisons-and-conditionals-srcagtensorzig)).
- Constructors are the f32 branch's; the *initialization* entry points on
  `ExecContext` they delegate to (`fromSlice`, `fromBorrowedSlice`,
  `fromStorageSlice`, `fromBorrowedStorageSlice`, `empty`, `zeros`, `ones`,
  `full`, `scalar`, each taking a leading comptime `DType`) are catalogued
  in [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md).

```zig
test "arange linspace and seed-deterministic random constructors" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var r = try fucina.Tensor(.{.d}).arange(&ctx, 0, 2, 0.5); // end exclusive
    defer r.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.5, 1, 1.5 }, try r.dataConst());
    var l = try fucina.Tensor(.{.d}).linspace(&ctx, 0, 1, 3); // end INCLUSIVE
    defer l.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.5, 1 }, try l.dataConst());

    // Same seed → the same tensor (the §6.8 counter-stream contract).
    const M = fucina.Tensor(.{ .row, .col });
    var a = try M.randn(&ctx, .{ 2, 4 }, 42);
    defer a.deinit();
    var b = try M.randn(&ctx, .{ 2, 4 }, 42);
    defer b.deinit();
    try std.testing.expectEqualSlices(f32, try a.dataConst(), try b.dataConst());

    var g = try M.gumbel(&ctx, .{ 2, 4 }, 42); // finite Gumbel(0, 1) noise
    defer g.deinit();
    for (try g.dataConst()) |v| try std.testing.expect(std.math.isFinite(v));

    var identity = try M.eye(&ctx, 2);
    defer identity.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, try identity.dataConst());

    // i64-branch seed-stream constructors: randint over [low, high),
    // randperm as a rank-1 permutation of 0..n-1.
    var ints = try fucina.Tensor(.{ .dtype = .i64, .tags = .{ .row, .col } }).randint(&ctx, .{ 2, 4 }, 7, -3, 5);
    defer ints.deinit();
    for (try ints.dataConst()) |v| try std.testing.expect(v >= -3 and v < 5);
    var perm = try fucina.Tensor(.{ .dtype = .i64, .tags = .{.i} }).randperm(&ctx, 6, 7);
    defer perm.deinit();
    var total: i64 = 0;
    for (try perm.dataConst()) |v| total += v;
    try std.testing.expectEqual(@as(i64, 15), total); // 0+1+…+5
}
```

Error conditions (all recoverable errors, no panics):

| Error | Condition |
|---|---|
| `TensorError.InvalidShape` | raw rank ≠ `tensor_rank` in `variable`/`constant` (for `Tensor(.{})`: value not single-element); any zero dimension; rank 0 or > 8 |
| `TensorError.InvalidDataLength` | `values.len` ≠ product of `raw_shape` |
| `error.OutOfMemory` | allocation failure |

```zig
test "variable wraps a ctx-produced raw tensor" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .in })
        .variable(&ctx, try ctx.fromSlice(.f32, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 }));
    defer x.deinit();
    try std.testing.expect(x.requiresGrad());
    try std.testing.expectEqual(@as(usize, 3), x.dim(.in));
    try std.testing.expectEqual([2]usize{ 2, 3 }, x.shape());
    try std.testing.expectError(error.MutableDataRequiresNoGrad, x.data());
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6 }, try x.dataConst());
}
```

```zig
test "constant constructors" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var z = try fucina.Tensor(.{ .row, .col }).zeros(&ctx, .{ 2, 2 });
    defer z.deinit();
    var o = try fucina.Tensor(.{ .row, .col }).ones(&ctx, .{ 2, 2 });
    defer o.deinit();
    var f = try fucina.Tensor(.{ .row, .col }).full(&ctx, .{ 2, 2 }, 0.5);
    defer f.deinit();
    var u = try fucina.Tensor(.{ .row, .col }).empty(&ctx, .{ 2, 2 }); // uninitialized
    defer u.deinit();
    var s = try fucina.Tensor(.{}).scalar(&ctx, 3.5);
    defer s.deinit();
    try std.testing.expect(!a.requiresGrad());
    try std.testing.expectEqual(@as(f32, 3.5), try s.item());
    try std.testing.expectEqual([1]usize{1}, s.shape()); // scalar = raw rank-1 [1]
    try std.testing.expectError(error.InvalidDataLength, fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2 }));

    // *Like: same type (tags + dtype), shape from the receiver.
    var zl = try a.zerosLike(&ctx);
    defer zl.deinit();
    var mask = try a.fullLike(&ctx, -std.math.inf(f32));
    defer mask.deinit();
    try std.testing.expectEqual(a.shape(), zl.shape());
    try std.testing.expect(std.math.isNegativeInf((try mask.dataConst())[0]));
}
```

```zig
test "borrowed-storage constructors" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var backing = [_]f32{ 1, 2, 3, 4 };
    var t = try fucina.Tensor(.{ .row, .col }).fromBorrowedSlice(&ctx, .{ 2, 2 }, &backing);
    defer t.deinit();
    backing[0] = 42; // zero-copy: mutation is visible through the tensor
    try std.testing.expectEqual(@as(f32, 42), (try t.dataConst())[0]);

    const frozen = [_]f32{ 5, 6 }; // read-only source, no @constCast at the call site
    var c = try fucina.Tensor(.{.d}).fromBorrowedConstSlice(&ctx, .{2}, &frozen);
    defer c.deinit();
    try std.testing.expectEqual(@as(f32, 6), (try c.dataConst())[1]);
}
```

**Typed-constant branches** (int/bool and non-f32 float) share one
constructor set: `constant`, `fromTensor`, `fromSlice`,
`fromBorrowedConstSlice`, `empty`, `zeros`, `ones` — same semantics as the
f32 forms, elements typed `Scalar(dtype)`. There is no `full`, `scalar`, or
mutable `fromBorrowedSlice` on these branches. No `variable` except the
f16/bf16 leaf constructors `variable`/`variableFromSlice` ([§3.2](03-tensors-types-construction-and-data-access.md#32-the-four-facade-branches-srcagtensorzig)): gradients
are always f32.

**Block-quantized branch** constructors take *block* slices
(`Storage(dtype)`, e.g. `[]const fucina.quant.BlockQ8_0`):

```zig
pub fn constant(ctx: *ExecContext, value: RawTypedTensor) !Self
pub fn fromTensor(ctx: *ExecContext, value: RawTypedTensor) !Self
pub fn fromBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self // copies
pub fn fromStorageSlice(...) !Self       // alias of fromBlocks
pub fn fromBorrowedBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []Elem) !Self // borrows
```

`raw_shape` counts logical elements (so the innermost dimension must be a
multiple of the block size); `values` counts blocks. Quantized *formats* and
the packed-RHS machinery are [§10](10-quantization.md).

```zig
test "q8_0 constants: fromBlocks, dequantize, getRows" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const Q = fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
    const bits = struct {
        fn of(x: f32) u16 {
            return @bitCast(@as(f16, @floatCast(x)));
        }
    }.of;
    var blocks = [_]fucina.quant.BlockQ8_0{
        .{ .d = bits(1), .qs = [_]i8{1} ** fucina.quant.q8_0_block_size },
        .{ .d = bits(2), .qs = [_]i8{3} ** fucina.quant.q8_0_block_size },
    };
    var q = try Q.fromBlocks(&ctx, .{ 2, fucina.quant.q8_0_block_size }, &blocks);
    defer q.deinit();

    var dense = try q.to(&ctx, .f32); // dequantize; .f32 is the only target
    defer dense.deinit();
    try std.testing.expectEqual(@as(f32, 6), (try dense.dataConst())[fucina.quant.q8_0_block_size]);

    var row = try q.getRows(&ctx, .out, &.{1}, .batch); // dequantizing row gather
    defer row.deinit();
    comptime std.debug.assert(@TypeOf(row).dtype == .f32);
    try std.testing.expectEqual(@as(usize, 1), row.dim(.batch));
}
```

## 3.4 `deinit`, lifetime, and exec scopes (`src/ag/tensor.zig`, `src/tensor.zig`)

```zig
pub fn deinit(self: *Self) void
```

`deinit` releases one reference on the underlying refcounted buffer and
(f32 and typed-float branches) destroys the tensor's `GradState`; then sets
`self.* = undefined`,
so a second `deinit` on the same value is illegal (checked-UB in safe
builds, not a recoverable error). Buffer release is the driver of buffer-pool
recycling — the `defer x.deinit()` idiom returns transient storage to the
pool mid-forward (see [MEMORY-MODEL.md](../MEMORY-MODEL.md) and [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)).

- **Views retain.** Every view op ([§3.7](03-tensors-types-construction-and-data-access.md#37-views-and-structural-ops-srcagtensorzig-srctag_opszig-srcexecgather_scatterzig)) bumps the source buffer's refcount;
  the storage is freed only when the last owner — parent or view — deinits.
  View lifetimes are independent of their parents'.
- **Exec scopes.** While `ctx.openExecScope()` is active (the training
  pattern, [§5](05-automatic-differentiation.md)/[§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)), op *results* are adopted by the scope: the returned
  struct is a borrow with `scope_owned = true` and its `deinit` is a safe
  no-op — the scope releases value and graph node at `closeExecScope`. This
  lets the same defer-deinit forward code run scoped (training) and unscoped
  (inference). Tensors built by the *constructors* above are never
  scope-owned; only op results are.
- **No public clone.** `detach` and `materialize` are the copy/alias
  entry points (below); `grad()` returns an owned clone of the gradient.
- **Thread safety.** An `ExecContext` and the tensors flowing through it are
  single-threaded state: run ops on one context from one thread (parallelism
  happens *inside* ops via the context's worker pool, [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)). Gradient
  *accumulation* is internally mutex-guarded (`GradState.grad_mutex`), but
  facade tensor values carry no cross-thread handle synchronization. A
  backend-internal accelerator completion on storage is a host-visibility
  fence, not permission to share a handle across threads ([§9.9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#99-gpu-offload-srcbackendgpuzig-metalzig-cudazig)).

```zig
pub fn detach(self: *const Self, ctx: *ExecContext) !Self       // every scalar-dtype branch (a no-grad view)
pub fn materialize(self: *const Self, ctx: *ExecContext) !Self  // all branches
pub fn contiguous(self: *const Self, ctx: *ExecContext) !Self   // f32 branch only
pub fn isContiguous(self: *const Self) bool                     // all branches
```

`detach` returns a no-grad tensor **sharing storage** with `self` (a
refcounted view) — the values are live, the graph link is dropped.
`materialize` returns a **contiguous copy** in the tensor's logical order;
on the f32 branch it is differentiable (identity VJP through the strided
view). Use it to make a permuted/broadcast view exportable via `dataConst`.
`contiguous` is the borrow-if-contiguous variant (torch.contiguous): an
already-contiguous tensor returns a **zero-copy alias** of the same storage
(graph-linked through an identity VJP; in-place mutation of either handle is
visible through both), a strided view returns `materialize(ctx)` — an
independent snapshot. Either way the result is caller-owned (`deinit` it),
contiguous, and `dataConst`-safe; use `materialize` when a guaranteed copy
is wanted. `isContiguous` is the predicate behind that branch — true when
storage is dense row-major in logical order (innermost stride 1), i.e. the
layout `data`/`dataConst` require; false for strided views (permutes,
broadcasts, inner narrows).

## 3.5 Data access (`src/ag/tensor.zig`, `src/tensor.zig`)

```zig
pub fn item(self: *const Self) !f32                 // f32; typed: !Scalar(dtype); absent on quantized
pub fn data(self: *Self) ![]f32                     // mutable element view
pub fn dataConst(self: *const Self) ![]const f32    // read-only element view
pub fn copyTo(self: *const Self, dst: []f32) !void  // stride-aware copy out
pub fn asRawTensor(self: *const Self) *const RawTensor
```

(Element types are `Scalar(dtype)` on typed branches and the block struct
`Storage(dtype)` on the quantized branch.)

- `item` requires a single-element tensor (`len() == 1`, any shape of
  all-ones); otherwise `error.InvalidShape`. It is how scalar losses are
  read out (`try loss.item()`).
- `data`/`dataConst` return a slice over the tensor's storage. Both require
  a **contiguous** layout and fail with `error.UnsupportedView` on strided
  views (permutes, broadcasts, inner narrows) — `materialize` first, or use
  `copyTo`. On the f32 branch, `data` additionally fails with
  `error.MutableDataRequiresNoGrad` on a `requiresGrad()` tensor: graph
  values must not be mutated behind autograd's back. Writes through `data`
  on shared (viewed) storage are visible to every alias.
- `copyTo` copies the logical elements row-major into `dst`
  (`dst.len` must equal the storage length, else
  `error.InvalidDataLength`); on scalar dtypes it walks strides, so it works
  on non-contiguous views.
- `asRawTensor` exposes the underlying raw tensor pointer **read-only** —
  the escape hatch for shape/stride introspection and for interop with
  exec-layer entry points ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)/[§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)). Treat it strictly as a borrow: never
  `deinit` through it, never mutate, and never outlive the facade value.

```zig
test "views, contiguity, materialize, copyTo" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var m = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer m.deinit();
    var t = try m.permuteTo(&ctx, .{ .col, .row }); // zero-copy strided view
    defer t.deinit();
    try std.testing.expectError(error.UnsupportedView, t.dataConst());
    var tm = try t.materialize(&ctx); // contiguous copy in the new order
    defer tm.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, try tm.dataConst());

    var mid = try m.narrow(&ctx, .col, 1, 2); // view into m's storage
    defer mid.deinit();
    var buf: [4]f32 = undefined;
    try mid.copyTo(&buf); // stride-aware; works on non-contiguous views
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 5, 6 }, &buf);
}
```

## 3.6 Shape and tag introspection (`src/ag/tensor.zig`, `src/tags.zig`)

```zig
pub fn shape(self: *const Self) [tensor_rank]usize   // runtime sizes, tag order
pub fn dim(self: *const Self, comptime tag: Tag) usize
pub fn axis(comptime tag: Tag) usize                 // comptime tag → axis index
pub fn hasTag(comptime tag: Tag) bool                // comptime membership test
```

`axis` is a compile error for an unknown tag (`tagIndexOrCompileError`);
`hasTag` is the non-failing probe generic code uses before calling `axis`.
`dim(tag)` is `shape()[axis(tag)]`. The comptime constants `axis_tags`,
`tag_count`, `tensor_rank`, `dtype` ([§3.1](03-tensors-types-construction-and-data-access.md#31-the-tensorspec-type-constructor-srcagtensorzig-srctagszig)) complete the introspection
surface. The tag algebra itself — how ops compute *result* tags — is [§7](07-named-axes-the-tag-algebra.md).

## 3.7 Views and structural ops (`src/ag/tensor.zig`, `src/tag_ops.zig`, `src/exec/gather_scatter.zig`)

Two families. **Zero-copy views** re-describe existing storage (shape/stride
arithmetic plus a refcount retain — nothing is moved); **copying ops**
produce new storage. On the f32 branch every one of these is differentiable
(each records a backward node routing gradients through the inverse
transform; mechanics in [§5](05-automatic-differentiation.md)); on constant branches the same names exist where
listed in [§3.10](03-tensors-types-construction-and-data-access.md#310-facade-surface-index) and simply produce constants.

### Zero-copy views

```zig
pub fn withTags(self, ctx, comptime new_tags_spec) !Tensor(...)   // rename only, same rank
pub fn alignTo(self, ctx, comptime target_tags_spec) !Tensor(...) // permute + insert missing tags as size-1 axes
pub fn permuteTo(self, ctx, comptime target_tags_spec) !Tensor(...) // pure permutation (same tag set)
pub fn transpose(self, ctx, comptime target_tags_spec) !Tensor(...) // alias of permuteTo
pub fn insertAxis(self, ctx, comptime tag, comptime axis_index) !Tensor(...) // new size-1 axis
pub fn squeeze(self, ctx, comptime tag) !Tensor(...)              // drop a size-1 axis
pub fn split(self, ctx, comptime tag, comptime split_tags_spec, split_shape) !Tensor(...)
pub fn merge(self, ctx, comptime out_tag, comptime merge_tags_spec) !Tensor(...)
pub fn broadcastTo(self, ctx, comptime target_tags_spec, target_shape) !Tensor(...)
pub fn narrow(self, ctx, comptime tag, start: usize, length: usize) !Self
pub fn select(self, ctx, comptime tag, index: isize) !Tensor(...)   // one position, axis removed
pub fn viewWithStrides(self, ctx, comptime new_tags_spec, raw_shape, raw_strides) !Tensor(...)
```

- `withTags` requires the same rank; it is the bridge between numeric-tag
  (`._0`) tensors from generic loaders and named-tag model code.
- `permuteTo`/`transpose` require the same tag *set* (comptime-checked);
  `alignTo` additionally inserts missing target tags as size-1, stride-0
  axes.
- `split` factors one axis into several (`split_shape` product must equal
  the axis length, else `error.InvalidShape`); `merge` fuses *adjacent*
  axes (tags must be contiguous and in tensor order — comptime error
  otherwise; stride-incompatible layouts fail with
  `error.UnsupportedView`).
- `broadcastTo` produces stride-0 axes for missing or size-1 tags;
  mismatched non-1 sizes fail with `error.ShapeMismatch`.
- `squeeze` fails with `error.InvalidShape` if the axis size is not 1.
- `select` is torch.select / `x[i]`: one position of `tag` with the axis
  removed (the single-slice sibling of `unbindInto`; composed narrow →
  squeeze, so the result is a zero-copy view aliasing the selected row).
  `index` counts from the end when negative (torch convention);
  out-of-range errors with `IndexOutOfBounds`. Scope-required under
  gradients ([§5](05-automatic-differentiation.md)); the gradient is the exact scatter — unselected
  positions receive zero.
- `viewWithStrides` is the audited escape hatch for layouts no tag
  operation can express: explicit raw shape + strides with a new tag set,
  bounds-checked against the underlying buffer
  (`error.InvalidDataLength` when the view would overrun).

```zig
test "split and merge" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .seq, .qkv })
        .fromSlice(&ctx, .{ 2, 6 }, &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 });
    defer x.deinit();
    var heads = try x.split(&ctx, .qkv, .{ .head, .d }, .{ 2, 3 }); // [seq, head, d] view
    defer heads.deinit();
    try std.testing.expectEqual([3]usize{ 2, 2, 3 }, heads.shape());
    var back = try heads.merge(&ctx, .qkv, .{ .head, .d }); // [seq, qkv] again
    defer back.deinit();
    try std.testing.expectEqualSlices(f32, try x.dataConst(), try back.dataConst());
}
```

```zig
test "tag-directed axis views" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var row = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer row.deinit();
    var aligned = try row.alignTo(&ctx, .{ .batch, .d }); // missing tag => size-1 axis
    defer aligned.deinit();
    try std.testing.expectEqual([2]usize{ 1, 3 }, aligned.shape());
    var grid = try row.broadcastTo(&ctx, .{ .batch, .d }, .{ 2, 3 }); // stride-0 view
    defer grid.deinit();
    try std.testing.expectEqual(@as(usize, 2), grid.dim(.batch));

    var col = try row.insertAxis(&ctx, .b, 0); // [b=1, d]
    defer col.deinit();
    var flat = try col.squeeze(&ctx, .b); // back to [d]
    defer flat.deinit();
    var renamed = try flat.withTags(&ctx, .{.feature}); // pure re-tag, same storage
    defer renamed.deinit();
    try std.testing.expectEqual(@as(usize, 3), renamed.dim(.feature));
}
```

```zig
test "viewWithStrides escape hatch" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var v = try fucina.Tensor(.{.flat}).fromSlice(&ctx, .{4}, &.{ 1, 2, 3, 4 });
    defer v.deinit();
    var m = try v.viewWithStrides(&ctx, .{ .row, .col }, .{ 2, 2 }, .{ 2, 1 });
    defer m.deinit();
    try std.testing.expectEqual(@as(f32, 3), (try m.dataConst())[2]);
}
```

### Copying structural ops

```zig
pub fn concat(self, ctx, comptime tag, others: []const *const Self) !Self
pub fn gather(self, ctx, comptime tag, indices: []const usize, comptime out_tag) !Tensor(...)
pub fn indexSelect(self, ctx, comptime tag, indices, comptime out_tag) !Tensor(...)
pub fn setSlice(self, ctx, comptime tag, start: usize, update: *const Self) !Self
pub fn setRows(self, ctx, comptime tag, indices: []const usize, update: *const Self) !Self
pub fn flatten(self, ctx, comptime out_tag) !Tensor(.{out_tag})
pub fn pad(self, ctx, comptime tag, before: usize, after: usize, fill: f32) !Self
pub fn zeroPad2d(self, ctx, comptime h_tag, comptime w_tag, padding: anytype) !Self
pub fn constantPad2d(self, ctx, comptime h_tag, comptime w_tag, padding: anytype, fill: f32) !Self
pub fn zeroSlice(self, ctx, comptime tag, start: usize, length: usize) !Self
pub fn zeroRows(self, ctx, comptime tag, indices: []const usize) !Self
pub fn flip(self, ctx, comptime tag) !Self
pub fn roll(self, ctx, comptime tag, shift: isize) !Self
pub fn repeatAxis(self, ctx, comptime tag, n: usize) !Self
pub fn stack(self, ctx, comptime new_tag, comptime axis_index, others: []const *const Self) !Tensor(...)
pub fn unbindInto(self, ctx, comptime tag, out: []Tensor(...)) !void
pub fn maskedSelect(self, ctx, mask, comptime out_tag) !Tensor(.{out_tag})
pub fn maskedScatter(self, ctx, mask, comptime values_tag, values: *const Tensor(.{values_tag})) !Self
pub fn rollBy(self, ctx, comptime tag, offsets: []const isize) !Self
pub fn shiftBy(self, ctx, comptime tag, offsets: []const isize, fill: f32) !Self
pub fn reshape(self, ctx, comptime new_tags_spec, new_shape: [...]usize) !Tensor(normalizeTags(new_tags_spec))
pub fn sliceStep(self, ctx, comptime tag, start: usize, length: usize, step: usize) !Self
pub fn slice(self, ctx, spec) !Self  // multi-axis basic slicing over a per-tag range struct
pub fn diagonal(self, ctx, comptime tag_a, comptime tag_b, comptime out_tag) !Tensor(...)
pub fn trace(self, ctx, comptime tag_a, comptime tag_b) !Tensor(...)
pub fn diag(self, ctx, comptime out_tags_spec) !Tensor(normalizeTags(out_tags_spec))
pub fn nonzero(self, allocator: std.mem.Allocator) ![]usize
pub fn indexAdd(self, ctx, comptime tag, indices: []const usize, update: *const Self) !Self
pub fn takeAlongAxis(self, ctx, comptime tag, indices) !Self
pub fn scatterAdd(self, ctx, comptime tag, indices, src: *const Self) !Self
pub fn scatter(self, ctx, comptime tag, indices, src: *const Self) !Self
```

- `concat` joins along an existing tag (all other dims must match); the
  result owns fresh storage. `gather` copies the rows selected by `indices`
  along `tag`, re-tagging that axis `out_tag` (`out_tag` may equal `tag`);
  out-of-range indices error with `IndexOutOfBounds`. `indexSelect` is
  torch.index_select — `gather` with a rank-1 **i64** index tensor (the
  argmax/topK/sort convention; other dtypes are compile errors), read
  host-side into the same `[]usize` path; entries outside `[0, dim(tag))`
  error with `IndexOutOfBounds` (no wrapping), duplicate reads accumulate
  their gradients, and the index tensor is control data outside the graph.
- `setSlice`/`setRows` are *functional* scatter updates: they return a copy
  of `self` with the range `[start, start+len)` / the given rows along
  `tag` overwritten by `update`; the originals are untouched. Gradients
  flow to both `self` (masked) and `update`. `setRows` requires unique
  in-range indices (`IndexOutOfBounds` / `InvalidShape` on duplicates), and
  `update` must match `self` except along `tag`, where it must have
  `indices.len` rows. `zeroSlice`/`zeroRows` are the fill-with-zero variants
  (no `update` operand).
- `flatten` reshapes to rank-1 under a new tag, materializing first only if
  the source is non-contiguous.
- `pad` grows one axis by `before + after` positions holding `fill`;
  `flip` reverses an axis; `roll` rotates it by `shift` (negative allowed);
  `repeatAxis` tiles the axis `n` times (`n == 0` errors with
  `InvalidShape`, `n == 1` is a zero-copy identity view).
- `zeroPad2d`/`constantPad2d` are torch nn.ZeroPad2d/nn.ConstantPad2d over
  named axes: `padding` is an integer (all four sides) or a 4-tuple/array
  in the torch order `(left, right, top, bottom)` — left/right grow
  `w_tag`, top/bottom grow `h_tag`; negative entries CROP that side (the
  F.pad constant-mode rule). Cropping an axis to zero size or below errors
  `InvalidShape` — the one deliberate divergence: torch returns an empty
  tensor at exactly zero (zero-size tensors are not representable here)
  and errors only below zero. Any rank carrying both tags works; pad
  positions drop their gradient and cropped source positions receive zero
  gradient. Forward/backward semantics are pinned against torch 2.0
  vectors (nn.ZeroPad2d / nn.ConstantPad2d / F.pad, gradients included).
- `stack` inserts a new axis on every input and concatenates along it
  (torch.stack); `unbindInto` fills a caller-provided slice with the
  `dim(tag)` sub-tensors of `self` with `tag` removed — the caller owns and
  deinits every filled entry (under an exec scope they are scope-owned
  borrows and `deinit` is a no-op); `out.len` must equal `dim(tag)`.
  `maskedSelect` returns the elements where `mask` is nonzero as a rank-1
  tensor. Selecting nothing errors with the dedicated `EmptySelection`
  (zero-size tensors are not representable) — distinct from the shape
  errors, so the data-dependent no-match outcome is catchable apart from
  caller bugs; pre-counting with a mask sum avoids the error path entirely
  (snippet below).
- `maskedScatter` is `maskedSelect`'s inverse (torch masked_scatter with an
  exact-count contract): it returns a copy of `self` with the rank-1
  `values` written into the nonzero-mask positions in row-major order.
  `values` must hold exactly `count(mask != 0)` elements (`InvalidShape`
  otherwise) and the mask must select at least one (`EmptySelection`
  otherwise, as in `maskedSelect`); the mask follows the
  `where`/`maskedFill` convention (non-grad `!= 0`, `self`'s shape,
  contiguous). Differentiable in `self` (grad zeroed where scattered) and
  `values` (grad gathered from the selected positions).
- `rollBy`/`shiftBy` generalize `roll` to one shift per *section* (the
  sub-vector obtained by fixing all axes except `tag`), keeping `roll`'s
  sign convention. `offsets` is host-side control data like `gather`
  indices: one `isize` per section, row-major over the remaining axes in
  tag order (`offsets.len == numel / dim(tag)`, else `InvalidShape`; a
  rank-1 tensor takes a single offset and `rollBy` matches `roll`). `rollBy`
  wraps (exact permutation gradient); `shiftBy` is non-circular — shifted-in
  positions hold the constant `fill` (no gradient) and shifted-out source
  positions receive zero gradient.
- `reshape` is torch.reshape over named axes: an arbitrary row-major
  reinterpretation to `new_tags_spec`/`new_shape` (element counts must
  match, `InvalidShape` otherwise) with the torch view-or-materialize
  rule — a contiguous source stays a zero-copy view, a strided one
  materializes first (composed flatten → split; a rank-1 target
  degenerates to plain `flatten`).
- `sliceStep` is `narrow` with a step (torch `x[start::step]` on one
  axis): a zero-copy strided view on no-grad tensors; under gradients it
  lowers to `gather` over the stepped indices (a copy with the exact
  scatter-add record — the `flip`/`roll` precedent).
- `slice` is multi-axis basic slicing (torch/numpy `x[1:-1, ::2]`,
  positive steps): `spec` is a struct literal naming the tags to slice —
  `.{ .h = .{ .start = 1, .end = -1 }, .w = .{ .step = 2 } }` — each
  field a `fucina.SliceRange`-shaped range (`start`/`end`/`step`, each
  optional); unnamed axes pass through whole, and naming a tag not on the
  tensor is a compile error. torch bounds semantics: negatives count from
  the end, `end = null` means the axis dim, out-of-range bounds clamp;
  `step == 0` or an empty result error with `InvalidShape` (zero-size
  tensors are not representable). Negative steps are deliberately
  unsupported (torch rejects them in basic indexing too; strides are
  unsigned, so a reversed view cannot exist — compose `flip`). Lowered to
  per-axis `narrow`/`sliceStep` in tag order: step-1 ranges stay zero-copy
  views with exact scatter gradients; stepped axes follow the `sliceStep`
  contract. Scope-required under gradients when more than one axis is
  sliced.
- `diagonal` views the main diagonal over a (`tag_a`, `tag_b`) plane
  (torch.diagonal, offset 0, any rank carrying both tags): length
  `min(dim_a, dim_b)`, both tags removed, the diagonal appended LAST as
  `out_tag`; zero-copy and differentiable. `trace` is composed diagonal →
  sum; `diag` embeds a rank-1 tensor as the diagonal of an `[n, n]`
  matrix (composed zeros → setRows → reshape).
  `diagEmbed(ctx, vec_tag, .{ row_tag, col_tag })` is the BATCHED embed
  (torch.diag_embed generalized to named axes): at any rank carrying
  `vec_tag`, `out[…, i, j] = self[…, i]·[i == j]` with `vec_tag` renamed
  to `row_tag` in place and `col_tag` appended LAST — composed rename →
  broadcast-multiply against an `eye` constant, so the gradient is the
  exact diagonal extraction.
- `nonzero` returns the row-major flat indices of the nonzero elements
  (NaN counts) as a HOST `[]usize` the caller frees — data-dependent
  cardinality stays host-side by design (ARCHITECTURE.md), so a no-match
  result is just an empty slice, and the indices feed straight into
  `gather`/`setRows`/`indexAdd`/`oneHot`.
- `indexAdd` is torch.index_add: a copy of `self` with `update`'s rows
  ADDED at host-side `indices` along `tag` — unlike `setRows` duplicates
  are allowed and accumulate; differentiable in both (identity /
  row-gather).
- `takeAlongAxis`/`scatterAdd`/`scatter` are the per-ELEMENT indexed ops
  (torch gather / scatter_add / scatter): the index operand is a
  same-tagged i64 tensor in the argmax/topK/sort index convention, so
  selection-op outputs feed them directly ([§4.16](04-tensor-operations.md#416-selection-argmax-topk-sort-routertopk-srcagtensorzig-srcexectopkzig)-17).
- `unbindInto`, `select`, `slice` (more than one sliced axis),
  `maskedSelect`, `maskedScatter`, `rollBy`, `shiftBy`, `stack`,
  `zeroPad2d`, `constantPad2d`, `reshape` (multi-tag targets), `trace`,
  `diag`, and `diagEmbed` are *composed* ops: when gradients are tracked
  they require an active exec scope and error with
  `error.ActiveExecScopeRequired` otherwise ([§5](05-automatic-differentiation.md)).
- Quantized branch: `concat` (rank-2, row axis only) and
  `getRows(ctx, tag, indices, out_tag)` — a fused gather+dequantize
  returning an **f32** tensor (see the snippet in [§3.3](03-tensors-types-construction-and-data-access.md#33-construction-and-ownership-srcagtensorzig-srcexeczig)); both comptime-reject
  other configurations.

```zig
test "maskedSelect no-match outcome: count first or catch EmptySelection" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, -2, 3 });
    defer x.deinit();
    var mask = try x.compare(&ctx, .gt, 10); // nothing matches
    defer mask.deinit();

    // Count-first idiom: the .bool mask sums to the selection count (i64).
    var count = try mask.sumAll(&ctx);
    defer count.deinit();
    try std.testing.expectEqual(@as(i64, 0), try count.item());

    // Or catch the dedicated error — EmptySelection is the recoverable
    // no-match outcome; shape errors (caller bugs) stay loud.
    var picked = x.maskedSelect(&ctx, mask, .m) catch |err| switch (err) {
        error.EmptySelection => null,
        else => return err,
    };
    defer if (picked) |*p| p.deinit();
    try std.testing.expect(picked == null);
}
```

```zig
test "zeroPad2d follows the torch (left, right, top, bottom) order" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .h, .w }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();

    // nn.ZeroPad2d((1, 0, 0, 1)): one zero column left, one zero row below.
    var padded = try x.zeroPad2d(&ctx, .h, .w, .{ 1, 0, 0, 1 });
    defer padded.deinit();
    try std.testing.expectEqualSlices(f32, &.{
        0, 1, 2,
        0, 3, 4,
        0, 0, 0,
    }, try padded.dataConst());

    // Negative padding crops (the F.pad constant-mode rule).
    var cropped = try x.zeroPad2d(&ctx, .h, .w, .{ -1, 0, 0, 0 });
    defer cropped.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4 }, try cropped.dataConst());
}
```

```zig
test "concat, gather, setSlice, setRows" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 1, 2 }, &.{ 5, 6 });
    defer b.deinit();
    var cat = try a.concat(&ctx, .row, &.{&b}); // [3, 2]
    defer cat.deinit();
    try std.testing.expectEqual(@as(usize, 3), cat.dim(.row));

    var picked = try cat.gather(&ctx, .row, &.{ 2, 0 }, .sel); // rows 2 and 0
    defer picked.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 6, 1, 2 }, try picked.dataConst());

    var patch = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 1, 2 }, &.{ 9, 9 });
    defer patch.deinit();
    var replaced = try cat.setSlice(&ctx, .row, 1, &patch); // overwrite row 1
    defer replaced.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 9, 9, 5, 6 }, try replaced.dataConst());

    var scattered = try cat.setRows(&ctx, .row, &.{2}, &patch); // overwrite row 2
    defer scattered.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 9, 9 }, try scattered.dataConst());
}
```

The same structural surface works on typed constants:

```zig
test "integer constants: structural ops only" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var ids = try fucina.Tensor(.{ .dtype = .i64, .rank = 2 })
        .fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();
    comptime std.debug.assert(@TypeOf(ids).axis_tags[0] == ._0); // auto numeric tags
    var named = try ids.withTags(&ctx, .{ .batch, .seq });
    defer named.deinit();
    var t = try named.transpose(&ctx, .{ .seq, .batch });
    defer t.deinit();
    var out: [6]i64 = undefined;
    try t.copyTo(&out);
    try std.testing.expectEqualSlices(i64, &.{ 1, 4, 2, 5, 3, 6 }, &out);
    comptime std.debug.assert(@hasDecl(@TypeOf(ids), "add")); // wrapping int math (§4.19)
    comptime std.debug.assert(!@hasDecl(@TypeOf(ids), "softmax")); // float NN ops stay off ints
}
```

## 3.8 Casting: `to(dtype)` (`src/ag/tensor.zig`, `src/exec/convert.zig`)

```zig
pub fn to(self: *const Self, ctx: *ExecContext, comptime target_dtype: DType)
    !Tensor(.{ .dtype = target_dtype, .tags = axis_tags })
```

`to` always copies (a new tensor; the source is untouched). Supported
conversions per branch:

| Source branch | Targets | Notes |
|---|---|---|
| f32 | any scalar dtype | `to(.f32)` is a differentiable copy; `to(.f16)`/`to(.bf16)` are DIFFERENTIABLE narrows (the mixed-precision seam, [§5.1](05-automatic-differentiation.md#51-the-gradient-model-srcagtensorzig-srcagcorezig): the backward is the identity on the f32 upstream gradient); every other target requires no-grad and fails with `error.GradientCastUnsupported` on a `requiresGrad()` tensor — `to(.f64)` is a float↔float cast, non-float targets follow the `castScalar` semantics in the int/bool row |
| f16/bf16 | any scalar dtype | `to(.f32)` is a DIFFERENTIABLE widen when the source requires grad (the f32 gradient flows back unchanged); casts to any non-f32 target require no-grad (`error.GradientCastUnsupported`) — float targets are float↔float casts, non-float targets follow the `castScalar` semantics in the int/bool row |
| f64 constant | any scalar dtype | always no-grad (an f64 constant never carries a gradient); float targets are float↔float casts, non-float targets follow the `castScalar` semantics in the int/bool row |
| int/bool constant | any scalar dtype | no-grad `castScalar` semantics: integer↔integer WRAPS (two's complement); integer→float is exact where representable; float→integer truncates toward zero and SATURATES at the target bounds with NaN → 0; anything→bool is `!= 0` (NaN → true); bool→number is 0/1 |
| block-quantized | `.f32` only | dequantization; other targets are a compile error |

```zig
test "f16 constants: forward math, reductions widen" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const H = fucina.Tensor(.{ .dtype = .f16, .tags = .{ .m, .k } });
    var a = try H.fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var twice = try a.add(&ctx, &a);
    defer twice.deinit();
    comptime std.debug.assert(@TypeOf(twice).dtype == .f16); // pointwise keeps dtype
    var s = try a.sum(&ctx, .k, .{});
    defer s.deinit();
    comptime std.debug.assert(@TypeOf(s).dtype == .f32); // reductions widen to f32
    var dense = try a.to(&ctx, .f32);
    defer dense.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, try dense.dataConst());
}
```

```zig
test "detach and cast rules" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variable(&ctx, try ctx.fromSlice(.f32, &.{2}, &.{ 1, 2 }));
    defer x.deinit();
    var frozen = try x.detach(&ctx); // shares storage, drops grad tracking
    defer frozen.deinit();
    try std.testing.expect(!frozen.requiresGrad());
    try std.testing.expectError(error.InvalidShape, frozen.item()); // not single-element
    try std.testing.expectError(error.GradientCastUnsupported, x.to(&ctx, .f64));
    var narrowed = try x.to(&ctx, .f16); // differentiable narrow: stays in the graph
    defer narrowed.deinit();
    try std.testing.expect(narrowed.requiresGrad());
    var half = try frozen.to(&ctx, .f16); // constants cast freely between floats
    defer half.deinit();
    comptime std.debug.assert(@TypeOf(half).dtype == .f16);
}
```

## 3.9 Gradient accessors (`src/ag/tensor.zig`, `src/ag/core.zig`; mechanics in [§5](05-automatic-differentiation.md))

f32-branch surface. `requiresGrad` also exists on every other branch:
hard-wired `false` on the scalar/quantized constants (never true on f64 —
no leaf can exist), a real
`grad_state != null` check on f16/bf16 — whose leaf-autograd accessors
(`variable`, `grad`, `gradView`, `zeroGrad`, `detach`) mirror the f32 ones
with f32-dtype gradient results ([§5.1](05-automatic-differentiation.md#51-the-gradient-model-srcagtensorzig-srcagcorezig)).

```zig
pub fn requiresGrad(self: *const Self) bool
pub fn backward(self: *const Self, ctx: *ExecContext) !void  // error.NoGradientGraph on constants
pub fn backwardWithGrad(self: *const Self, ctx: *ExecContext, grad_output: *const Self) !void // explicit output gradient
pub fn grad(self: *const Self, ctx: *ExecContext) !?Self     // owned CLONE of the gradient, or null
pub fn gradView(self: *const Self, ctx: *ExecContext) !?Self // refcounted VIEW of the live gradient
pub fn zeroGrad(self: *const Self) void                      // drop accumulated grad; no-op on constants
```

`grad`/`gradView` return `null` before any `backward` has produced a
gradient (and again after `zeroGrad`). Both returns are no-grad tensors the
caller must `deinit`; `gradView` shares storage with the accumulator *as of
that moment* — a later `backward` pass accumulates into a fresh private
buffer (the held view defeats copy-on-write, [§5.3](05-automatic-differentiation.md#53-reading-seeding-and-resetting-gradients-srcagtensorzig-srcagcorezig)), so the view keeps the
stale value; use `grad` to observe later passes. Training loops call
`zeroGrad` between steps so gradients
do not accumulate across them. Graph construction, `fucina.noGrad`,
checkpointing, and the traversal/seeding contract of `backward` and
`backwardWithGrad` (a non-scalar output needs the explicit output gradient;
one backward per graph) are [§5](05-automatic-differentiation.md).

```zig
test "grad accessors on the facade" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variable(&ctx, try ctx.fromSlice(.f32, &.{3}, &.{ 1, 2, 3 }));
    defer x.deinit();
    try std.testing.expect((try x.grad(&ctx)) == null); // no backward run yet
    var loss = try x.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var g = (try x.grad(&ctx)).?; // owned clone; caller deinits
    defer g.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1 }, try g.dataConst());
    x.zeroGrad(); // drop the accumulated gradient
    try std.testing.expect((try x.grad(&ctx)) == null);
}
```

## 3.10 Facade surface index

Complete public surface of `Tensor(spec)`, split by owning section. [§3](03-tensors-types-construction-and-data-access.md)
methods are documented above; [§4](04-tensor-operations.md) covers every math/NN op in depth.

**f32 branch — [§3](03-tensors-types-construction-and-data-access.md) (construction / lifetime / data / structure):**
`variable`, `variableFromSlice`, `constant`, `fromTensor`, `fromSlice`,
`fromBorrowedSlice`, `fromBorrowedConstSlice`, `empty`, `zeros`, `ones`,
`full`, `scalar`, `emptyLike`, `zerosLike`, `onesLike`, `fullLike`,
`arange`, `linspace`, `oneHot`, `eye`, `rand`, `uniform`, `randn`, `normal`,
`bernoulli`, `gumbel`,
`deinit`, `asRawTensor`, `item`, `data`, `dataConst`,
`copyTo`, `detach`, `materialize`, `contiguous`, `isContiguous`,
`requiresGrad`, `zeroGrad`,
`backward`, `backwardWithGrad`, `grad`, `gradView`, `axis`, `hasTag`, `dim`,
`shape`, `to`,
`withTags`, `viewWithStrides`, `alignTo`, `permuteTo`, `transpose`,
`insertAxis`, `squeeze`, `split`, `merge`, `reshape`, `broadcastTo`,
`narrow`, `select`, `slice`, `sliceStep`,
`flatten`, `gather`, `indexSelect`, `maskedSelect`, `maskedScatter`, `nonzero`, `flip`,
`roll`, `rollBy`, `shiftBy`, `concat`, `stack`, `unbindInto`, `repeatAxis`,
`pad`, `zeroPad2d`, `constantPad2d`, `setSlice`, `setRows`, `indexAdd`,
`takeAlongAxis`, `scatterAdd`, `scatter`, `zeroSlice`,
`zeroRows`, `diagonal`, `diag`, `diagEmbed`, `trace`; consts
`axis_tags`, `tag_count`, `tensor_rank`, `dtype`; fields `value`,
`grad_state`, `scope_owned`.

**f32 branch — [§4](04-tensor-operations.md) (math / NN):**
`add`, `sub`, `mul`, `div`, `scale`, `addScalar`, `subScalar`, `divScalar`,
`powScalar`, `log1p`, `takeAddNoGrad`, `takeScaleNoGrad`,
`addAxisVectorInPlace`, `addAxisVectorUnaryInPlace`, `addScaledInPlace`,
`biasAdd`, `where`, `maskedFill`, `compare`, `logicalAnd`, `logicalOr`,
`logicalXor`, `logicalNot`, `unary`, `elementalUnary`, `elementalBinary`,
`relu`, `leakyRelu`, `exp`, `sqrt`,
`rsqrt`, `sigmoid`, `silu`, `log`, `neg`, `abs`, `sin`, `cos`, `tanh`,
`fastTanh`, `gelu`, `quickGelu`, `elu`, `geluErf`, `erf`,
`floor`, `ceil`, `round`, `sign`, `reciprocal`, `clamp`, `clampMin`, `clampMax`,
`maximum`, `minimum`, `pow`, `isnan`, `isinf`, `isfinite`,
`dropout`, `gated`, `glu`, `swiglu`, `geglu`, `situ`, `splitGated`, `sum`, `mean`,
`cumsum`, `prod`, `cumprod`, `segmentSum`, `linearRecurrence`, `variance`,
`standardizeAxis`, `sumAll`,
`sumMany`, `any`, `all`, `anyAll`, `allAll`, `norm`, `normAll`,
`bandPart`, `tril`, `triu`,
`logsumexp`, `logSoftmax`, `argmax`, `multinomial`,
`max`, `min`, `topK`, `sort`, `argsort`, `routerTopK`, `softmax`, `rmsNorm`,
`rmsNormMul`, `rmsNormMulAdd`, `rmsNormMulRopeHalfPrepared`, `layerNorm`,
`groupNorm`, `crossEntropy`, `crossEntropy`, `linearCrossEntropy`, `linearDistill`,
`mseLoss`, `huberLoss`,
`bceLoss`, `klDivLoss`, `nllLoss`, `l2Normalize`, `cosineSimilarity`,
`rope`, `matmul`, `dot`, `addDot`, `einsum`, `dotTernarySte`, `packRhs`, `dotPacked`,
`rmsNormMulDotPacked`,
`splitSwiGluDotPacked`, `gegluQuantDotPacked`, `groupedAttention`,
`conv2d`, `conv2dRelu`, `prepareConv2dWeights`, `conv2dPrepared`,
`conv2dPreparedRelu`, `maxPool2d`, `avgPool2d`, `upsample2xNearest`,
`unfold`, `fold`,
`prelu`, `channelAffine`, `relposShift`, `causalDepthwiseConv1d`,
`causalConv1d`, `groupedCausalConv1d`, `conv1d`, `convTranspose1d`,
`snake`. The root free function `fucina.einsumMany(ctx, out_tags, operands)`
is the N-ary companion of `einsum` ([§4.8](04-tensor-operations.md#48-dot-tag-directed-contraction-srcagtensorzig-srctag_opszig)).

**Shared view set** (every scalar-dtype branch, one implementation in
`src/ag/tensor/views.zig`; differentiable on f32, no-grad constants
elsewhere, where a grad-requiring operand is `error.UnsupportedGradient`):
`materialize`, `contiguous`, `detach`, `withTags`, `viewWithStrides`,
`alignTo`, `permuteTo`, `transpose`, `insertAxis`, `squeeze`, `split`,
`merge`, `reshape`, `broadcastTo`, `flatten`, `flip`, `roll`, `rollBy`,
`narrow`, `select`, `sliceStep`, `slice`, `gather`, `setSlice`, `setRows`,
`concat`, `stack`, `unbindInto`, `repeatAxis`.

**Typed scalar-constant branch** (`.bool`/ints): `constant`, `fromTensor`,
`fromSlice`, `fromBorrowedConstSlice`, `empty`, `zeros`, `ones`,
`emptyLike`, `zerosLike`, `onesLike`, `item`,
`data`, `dataConst`, `copyTo`, `axis`, `hasTag`, `deinit`, `asRawTensor`,
`requiresGrad`, `dim`, `shape`, `isContiguous`, and the shared view set
above — all [§3](03-tensors-types-construction-and-data-access.md) — plus `to` ([§3.8](03-tensors-types-construction-and-data-access.md#38-casting-todtype-srcagtensorzig-srcexecconvertzig)), the
integer forward math `add`, `sub`, `mul`, `maximum`, `minimum`,
`divTrunc`, `divFloor`, `rem`, `mod`, `bitAnd`, `bitOr`, `bitXor`,
`sum`, `sumAll` ([§4.19](04-tensor-operations.md#419-math-on-non-f32-tensors-srcagtensorzig); on `.bool` the arithmetic
entries are compile errors — `to` and the counting `sum`/`sumAll` apply),
integer `compare` ([§4.6](04-tensor-operations.md#46-masks-comparisons-and-conditionals-srcagtensorzig), exact at any magnitude), the i64-only
seed-stream constructors `randint`, `randperm` ([§3.3](03-tensors-types-construction-and-data-access.md#33-construction-and-ownership-srcagtensorzig-srcexeczig)), and — on `.bool`
only — the `bandMask` constructor ([§4.6](04-tensor-operations.md#46-masks-comparisons-and-conditionals-srcagtensorzig)) and the mask combinators
`logicalAnd`, `logicalOr`, `logicalXor`, `logicalNot`.

**Typed float branch** (`.f16`/`.bf16`/`.f64`): everything in the
scalar-constant branch's [§3](03-tensors-types-construction-and-data-access.md) list (the shared view set included), plus `requiresGrad` and
`zeroGrad` on every typed float dtype (on f64 no leaf can exist, so
`requiresGrad` is `false` and `zeroGrad` no-ops), plus — f16/bf16 only,
compile errors on f64 — the leaf-autograd surface `variable`,
`variableFromSlice`, `grad`, `gradView` (f32 gradients; [§5.1](05-automatic-differentiation.md#51-the-gradient-model-srcagtensorzig-srcagcorezig)), plus `to`
([§3.8](03-tensors-types-construction-and-data-access.md#38-casting-todtype-srcagtensorzig-srcexecconvertzig)) and the forward-only math `add`,
`sub`, `mul`, `div`, `sum`, `mean`, `sumAll`, `dot`, `scale`, `divScalar`.
**f16/bf16 only** (each computes
through f32 and narrows once; a compile error on f64 — [§4.19](04-tensor-operations.md#419-math-on-non-f32-tensors-srcagtensorzig)): `unary` and
the named unary aliases (`relu`, `exp`, `sqrt`, `rsqrt`, `sigmoid`, `silu`,
`log`, `log1p`, `neg`, `abs`, `sin`, `cos`, `tanh`, `fastTanh`, `gelu`,
`quickGelu`, `elu`, `geluErf`, `erf`, `floor`, `ceil`, `round`, `sign`,
`reciprocal`), `softcap`, `leakyRelu`, `clamp`, `addScalar`,
`subScalar`, `powScalar`, `maximum`, `minimum`, `gated`, `glu`, `swiglu`,
`geglu`, `softmax` (plain `.{}` options), `logSoftmax`, `rmsNorm`,
`rmsNormMul`, `layerNorm` (plain `.{}` options), `cumsum`, `cumprod`,
`where`, `maskedFill`, `compare`, `pad`, `einsum`, the widened
reductions `max`, `min`, `prod`, `variance`, `logsumexp` (f32 results,
[§8.3](08-data-types-storage-and-the-raw-tensor-layer-internal.md#83-float-computeoutput-dtype-policy-srcdtypezig)), `argmax` (i64 result, [§4.16](04-tensor-operations.md#416-selection-argmax-topk-sort-routertopk-srcagtensorzig-srcexectopkzig)), and load-time dense `packRhs`
([§4.9](04-tensor-operations.md#49-explicit-matmul-ternary-ste-and-packed-rhs-gemms-srcagtensorzig)/[§10.3](10-quantization.md#103-rhs-containers-and-packed-layouts-srcbackendquanttypeszig-srcexecquant_matmulzig)).

**Block-quantized branch:** `constant`, `fromTensor`, `fromBlocks`,
`fromStorageSlice`, `fromBorrowedBlocks`, `deinit`, `asRawTensor`, `data`,
`dataConst`, `copyTo`, `requiresGrad`, `axis`, `hasTag`, `dim`, `shape`,
`isContiguous`, `withTags`, `to`, `materialize`, `concat`, `getRows` — all
[§3](03-tensors-types-construction-and-data-access.md) — plus
`packRhs`, `packRhsAs` (packed matmul RHS containers; [§10](10-quantization.md), used by
`dotPacked` in [§4](04-tensor-operations.md)). The same root helper `fucina.PackedRhs(dtype)` names
the f32/f16/bf16 dense pack and each quantized `packRhs` return type ([§10](10-quantization.md)).
