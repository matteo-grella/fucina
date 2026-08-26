# 7. Named axes: the tag algebra

Fucina names tensor axes with **tags** — Zig enum literals such as `.batch`,
`.seq`, `.d_model` — and every axis-level decision (broadcasting, contraction,
reduction, permutation) is made by tag identity, never by axis position at the
call site. Tags are **comptime-only data**: there is no runtime tag
representation and no tagged tensor *type*. The single runtime tensor currency
is the raw tensor ([§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)); the public `Tensor(tags_spec)` facade ([§3](03-tensors-types-construction-and-data-access.md), [§4](04-tensor-operations.md)) carries
its tag tuple purely in the type and re-attaches result tags at comptime after
each op.

Two internal modules implement this:

- `src/tags.zig` — the pure comptime tuple algebra: spec normalization,
  lookup, uniqueness/subset constraints, and result-tag computation. Every
  function here runs at compile time and violations are compile errors.
- `src/tag_ops.zig` — the tag-semantics op library: functions that take
  comptime tag tuples plus `*const` raw tensors and return **owned** raw
  tensors (tag-directed views, tag-driven broadcasting, multi-axis reduction,
  and `taggedDot` lowering onto the ExecContext matmul/bmm kernels, [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)).

Neither module is re-exported at the public root (`src/fucina.zig`): users
consume these semantics through `Tensor` methods, and the autograd VJPs
(`ag/backward/`, [§5](05-automatic-differentiation.md)) call the same library directly on raw gradients. This
section is the semantics contract for the public surface and the reference for
the internal library. Snippets demonstrate the semantics through the public
facade.

## 7.1 Tags, tag specs, and normalization (`src/tags.zig`)

```zig
pub const Tag = @TypeOf(.tag);                    // the enum-literal type
pub const inserted_axis = std.math.maxInt(usize); // axis-map sentinel (§7.3)
```

Any enum literal is a `Tag`; two tags are equal iff their spellings are equal
(`tagEqual` compares `@tagName` strings at comptime). Tag names carry no
built-in meaning — `._0`…`._7` are ordinary tags that happen to be generated
for rank specs.

Everything that accepts axes accepts a **tag spec**, normalized by:

```zig
pub fn normalizeTags(comptime tags_spec: anytype) [tagSpecLen(tags_spec)]Tag
pub fn tagSpecLen(comptime tags_spec: anytype) usize
pub fn dtypeFromSpec(comptime tags_spec: anytype) DType   // defaults to .f32
pub fn isTensorSpec(comptime tags_spec: anytype) bool
pub fn isRankSpec(comptime tags_spec: anytype) bool
pub fn rankFromSpec(comptime rank_spec: anytype) usize
pub fn autoTags(comptime rank: usize) [rank]Tag           // ._0, ._1, ... ._7
```

| Spec form | Example | Normalizes to |
|---|---|---|
| Tag tuple | `.{ .batch, .d }` | the tuple itself |
| Integer rank | `3` | `autoTags(3)` = `.{ ._0, ._1, ._2 }` |
| Struct spec | `.{ .dtype = .u16, .tags = .{ .batch, .seq } }` | the `tags` field |
| Struct spec (rank) | `.{ .dtype = .i64, .rank = 2 }` | `autoTags(2)` |

`isTensorSpec` recognizes a non-tuple struct with a `dtype`, `tags`, or `rank`
field; a struct spec with neither `tags` nor `rank` is a compile error
(`"tensor dtype specs must include tags or rank"`). `dtypeFromSpec` reads the
optional `.dtype` field and defaults to `.f32` — this is how non-f32 typed
tensors get tags ([§3](03-tensors-types-construction-and-data-access.md)). `rankFromSpec` rejects negative ranks (`"tensor rank
must be non-negative"`), non-integer specs (`"tensor tags must be a tag tuple
or a comptime rank"`), and ranks above `max_rank = 8` (`src/tensor.zig`,
`"too many tensor tags"`).

The public facade exposes the normalized result as comptime type members:
`Tensor(spec).axis_tags`, `.tag_count`, `.tensor_rank`, plus per-tag lookup
`axis(tag)`, `hasTag(tag)`, and runtime `dim(tag)` / `shape()` ([§3](03-tensors-types-construction-and-data-access.md)).

```zig
test "tag specs and comptime introspection" {
    const M = fucina.Tensor(.{ .batch, .d }); // explicit tag tuple
    const R = fucina.Tensor(2); // rank spec: auto tags ._0, ._1
    const S = fucina.Tensor(.{}); // scalar: empty tag tuple, raw rank 1
    comptime {
        std.debug.assert(M.axis(.d) == 1); // axis position by tag
        std.debug.assert(M.hasTag(.batch) and !M.hasTag(.channel));
        std.debug.assert(R.axis(._1) == 1 and R.tag_count == 2);
        std.debug.assert(S.tag_count == 0 and S.tensor_rank == 1);
    }
}
```

## 7.2 Lookup, equality, and constraint helpers (`src/tags.zig`)

```zig
pub fn tagEqual(comptime a: anytype, comptime b: anytype) bool
pub fn tagsEqual(comptime a: anytype, comptime b: anytype) bool   // elementwise + length
pub fn tagIndex(comptime tags: anytype, comptime tag: anytype) ?usize
pub fn tagIndexOrCompileError(comptime tags: anytype, comptime tag: anytype) usize
pub fn validateUniqueTags(comptime tags: anytype) void
pub fn validateSameTagSet(comptime source_tags: anytype, comptime target_tags: anytype) void
pub fn rawRank(comptime tag_count: usize) usize   // 0 -> 1, else tag_count
```

- `tagIndex` returns the axis position of a tag within a tuple, or `null`;
  `tagIndexOrCompileError` fails compilation with `"tensor tag not found"`.
- `validateUniqueTags` enforces the global uniqueness invariant — a tag tuple
  never repeats a tag (`"duplicate tensor tag"`). `Tensor(spec)` validates
  this at type construction, so no public tensor type can carry duplicates.
- `validateSameTagSet` is the permutation precondition: same length
  (`"permutation requires the same rank"`) and same membership
  (`"permutation target must contain the same tags"`).
- `rawRank` maps the tag count to the raw tensor rank: the empty tag tuple
  (scalar) is stored as a rank-1 raw tensor of shape `{1}` — there are no
  rank-0 raw tensors ([§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)).

## 7.3 Tuple rewrites and axis maps (`src/tags.zig`)

These compute the *result* tag tuple of an op; the facade uses them directly
in return types, so shape errors in tag terms surface as compile errors at the
call site.

```zig
pub fn removeTag(comptime tags: anytype, comptime tag: Tag) [tags.len - 1]Tag
pub fn removeTags(comptime tags: anytype, comptime remove_tags: anytype) [tags.len - remove_tags.len]Tag
pub fn replaceTag(comptime tags: anytype, comptime old_tag: Tag, comptime new_tag: Tag) [tags.len]Tag
pub fn insertTagAt(comptime tags: anytype, comptime tag: Tag, comptime axis_index: usize) [tags.len + 1]Tag
pub fn splitTags(comptime tags: anytype, comptime tag: Tag, comptime split_tags: anytype) [tags.len + split_tags.len - 1]Tag
pub fn mergeTags(comptime tags: anytype, comptime out_tag: Tag, comptime merge_tags: anytype) [tags.len - merge_tags.len + 1]Tag
pub fn mergeStartAxis(comptime tags: anytype, comptime merge_tags: anytype) usize
pub fn reduceAxesDescending(comptime tags: anytype, comptime reduce_tags: anytype) [reduce_tags.len]usize
```

Constraints (all compile errors):

| Helper | Enforced constraint |
|---|---|
| `removeTag` / `removeTags` | removed tags must exist; `remove_tags` unique |
| `replaceTag` | `old_tag` must exist; `new_tag` must not already exist elsewhere (`"replacement tensor tag already exists"`) |
| `insertTagAt` | index ≤ rank (`"insert axis out of bounds"`); tag must be new (`"inserted tensor tag already exists"`); result ≤ `max_rank` |
| `splitTags` | split axis must exist; `split_tags` non-empty, unique, and each must be new or equal to the split tag (`"split output tag already exists"`); result ≤ `max_rank` |
| `mergeTags` / `mergeStartAxis` | `merge_tags` non-empty, unique, and must appear **contiguously and in order** in `tags` (`"merge tags must be contiguous and in tensor order"`; a run overflowing the end of the tuple reports `"merge tags must be contiguous"`); `out_tag` must not collide with a retained tag unless it is one of the merged tags (`"merge output tag already exists"`) |

`reduceAxesDescending` maps reduce tags to axis indices sorted descending, so
a multi-axis reduction can strip one axis at a time without invalidating the
remaining indices ([§7.8](07-named-axes-the-tag-algebra.md#78-split-merge-flatten-and-multi-axis-reduction-srctag_opszig)).

Axis-map helpers translate tag decisions into per-axis permutation vectors
consumed by the view machinery; the sentinel `inserted_axis` means "no source
axis — inject a size-1, stride-0 axis here":

```zig
pub fn identityAxes(comptime rank: usize) [rank]usize
pub fn alignAxes(comptime source_tags: anytype, comptime target_tags: anytype) [target_tags.len]usize
pub fn insertAxes(comptime rank: usize, comptime axis_index: usize) [rank + 1]usize
pub fn squeezeAxes(comptime rank: usize, comptime axis_index: usize) [rank - 1]usize
```

`alignAxes` requires the target tuple to be unique and a **superset** of the
source (`"target tags must include all source tags"`); each target position
maps to its source axis or to `inserted_axis`. These back the facade's
`withTags`, `alignTo`, `permuteTo`/`transpose`, `insertAxis`, and `squeeze`
([§4](04-tensor-operations.md)).

## 7.4 Result-tag computation: pointwise and dot (`src/tags.zig`)

```zig
pub fn pointwiseResultTags(comptime left_tags: anytype, comptime right_tags: anytype) [pointwiseResultLen(...)]Tag
pub fn pointwiseResultLen(comptime left_tags: anytype, comptime right_tags: anytype) usize
```

Pointwise result tags are the **union in operand order**: all left tags in
left order, then every right-only tag appended in right order. Operand order
therefore determines the physical layout of the materialized result
(`{.d}` + `{.batch, .d}` produces tags `{.d, .batch}`, not `{.batch, .d}`).
The union must fit `max_rank`.

For a contraction over `contract_tag`, the operands' tags partition into
three comptime classes:

- **batch tags** — present in both operands and not the contract tag
  (`dotBatchTags`/`dotBatchLen`), in left-operand order;
- **left free tags** — left-only, non-contract
  (`dotLeftFreeTags`/`dotLeftFreeLen`);
- **right free tags** — right-only, non-contract
  (`dotRightFreeTags`/`dotRightFreeLen`).

```zig
pub fn dotResultTags(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) [dotResultLen(...)]Tag
pub fn dotResultLen(comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag) usize
```

`dotResultTags` = batch ++ left free ++ right free. The contract tag must be
present in **both** operands (compile error otherwise), and the result must
fit `max_rank`. Contracting a vector against a vector yields the empty tuple —
a scalar tensor.

Three canonical **storage orders** describe, per operand, the tag order a
direct kernel expects; the einsum lowering underneath `taggedDot` ([§7.9](07-named-axes-the-tag-algebra.md#79-taggeddot-tag-directed-contraction-and-its-lowering-srctag_opszig))
then picks each operand's plain-vs-transposed orientation at runtime by
which aligned view is already contiguous (a transposed GEMM is free, a
materialized permutation costs a copy — at most one of transA/transB per
call, and a side no orientation can express materializes at most once).

```zig
// All: (comptime left_tags: anytype, comptime right_tags: anytype, comptime contract_tag: Tag)
pub fn dotLeftOrder(...) [left_tags.len]Tag        // batch ++ left_free ++ contract
pub fn dotRightOrder(...) [right_tags.len]Tag      // batch ++ contract ++ right_free
pub fn dotRightTransBOrder(...) [right_tags.len]Tag// batch ++ right_free ++ contract
```

The einsum generalization derives every axis role from tag membership alone
(shared vs private × kept vs dropped):

```zig
pub const EinsumPart = enum { batch, contract, left_free, right_free, left_summed, right_summed };
pub fn einsumClassOfLeft(comptime right_tags: anytype, comptime out_tags: anytype, comptime tag: Tag) EinsumPart
pub fn einsumClassOfRight(comptime left_tags: anytype, comptime out_tags: anytype, comptime tag: Tag) EinsumPart
pub fn einsumPartTags(comptime left_tags, right_tags, out_tags: anytype, comptime part: EinsumPart) [einsumPartLen(...)]Tag
pub fn einsumPartLen(comptime left_tags, right_tags, out_tags: anytype, comptime part: EinsumPart) usize
pub fn einsumValidate(comptime left_tags: anytype, comptime right_tags: anytype, comptime out_tags: anytype) void
```

`einsumPartTags` reports each part in the owning operand's axis order; the
shared parts (batch, contract) are reported in LEFT order, matching the
`dot*` convention. `einsumValidate` compile-errors on duplicate output tags,
an output tag missing from both operands, or a result rank past `max_rank`.
The set helpers `unionTags`/`unionTagsLen` (the first tuple followed by
the second's tags not already present — a membership set, deliberately not
capped by `max_rank`) and `intersectTags`/`intersectTagsLen` (tags of the
first tuple also present in the second, first-tuple order) support the
einsum lowering and are general-purpose.

## 7.5 The op library contract (`src/tag_ops.zig`)

Every runtime function below takes comptime tag tuples plus `*const` raw
tensors and returns an **owned** raw tensor: the caller must `deinit` it.
Results that are views (`align`/`permute`/`broadcast`/`split`/`merge`, and
`cloneView` fast paths) retain the source's underlying buffer, so the view
stays valid even after the source tensor value is deinitialized (buffer
refcounting, [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)) — but they share storage: writing through one aliases the
other. Functions hold no state of their own; allocation and kernel dispatch go
through the `*ExecContext` argument, whose concurrency contract applies ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)).
All failures are recoverable Zig errors; nothing in this layer panics.

Rank validation is shared:

```zig
pub fn validateTensorRank(comptime tensor_dtype: DType, comptime tags: anytype,
                          value: *const TensorOf(tensor_dtype)) !void
```

An empty tag tuple requires a scalar value (`value.isScalar()`, i.e.
`len() == 1`); otherwise the raw rank must equal `tags.len`. Violations return
`TensorError.InvalidShape`. (`RawTensor` is `src/tensor.zig`'s `Tensor`,
spelled `fucina.internal.RawTensor` in-tree; `TensorOf` is its dtype-generic
form — [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md).)

Runtime error summary for the whole library:

| Condition | Error |
|---|---|
| value rank ≠ tag count (or non-scalar with empty tags) | `TensorError.InvalidShape` |
| broadcast dim conflict (both ≠ 1 and unequal) | `TensorError.ShapeMismatch` |
| dot contract or batch dim mismatch | `TensorError.ShapeMismatch` |
| split factors don't multiply to the axis dim, or a zero factor | `TensorError.InvalidShape` |
| merge over stride-incompatible axes | `TensorError.UnsupportedView` |

## 7.6 Alignment, permutation, and broadcast views (`src/tag_ops.zig`)

```zig
pub fn alignTensorTo(comptime tensor_dtype: DType, comptime source_tags: anytype,
                     source: *const TensorOf(tensor_dtype),
                     comptime target_tags: anytype) !TensorOf(tensor_dtype)
pub fn permuteTensorTo(comptime source_tags: anytype, source: *const RawTensor,
                       comptime target_tags: anytype) !RawTensor
pub fn broadcastTensorTo(comptime tensor_dtype: DType, comptime source_tags: anytype,
                         source: *const TensorOf(tensor_dtype),
                         comptime target_tags: anytype,
                         target_shape: [target_tags.len]usize) !TensorOf(tensor_dtype)
```

`alignTensorTo` is the workhorse view: it reorders axes into `target_tags`
order and, for every target tag absent from the source, injects a **size-1,
zero-stride axis**. Zero-copy always. Comptime preconditions: target unique,
target ⊇ source, target ≤ `max_rank`. An empty target returns a `cloneView`
(only reachable for scalar sources, since a non-empty source cannot be a
subset of an empty target).

`permuteTensorTo` adds `validateSameTagSet`: same tags, same rank — a pure
axis permutation with no injection.

`broadcastTensorTo` aligns first, then expands the aligned view to
`target_shape`: axes of size 1 (including injected ones) stretch to any size
with stride 0; a non-1 axis whose dim differs from the target returns
`TensorError.ShapeMismatch`. Still zero-copy. An empty target with a non-empty
source is a compile error (`"scalar broadcast target cannot drop source
tags"`) — broadcasting never drops axes.

Facade equivalents ([§4](04-tensor-operations.md)): `alignTo`, `permuteTo`/`transpose`, `broadcastTo`
(differentiable, `BroadcastBackward`), plus `withTags` (relabel, same rank),
`insertAxis`, and `squeeze` built on the axis maps of [§7.3](07-named-axes-the-tag-algebra.md#73-tuple-rewrites-and-axis-maps-srctagszig).

```zig
test "alignTo reorders and injects singleton axes" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .d }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    var y = try x.alignTo(&ctx, .{ .d, .batch, .channel }); // .channel absent -> size-1 axis
    defer y.deinit();
    const shape = y.shape();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2, 1 }, &shape);

    var copied = [_]f32{0} ** 6;
    try y.copyTo(&copied); // transposed traversal of the same buffer
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 2, 5, 3, 6 }, &copied);
}
```

```zig
test "broadcastTo expands missing tags without copying" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var bias = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer bias.deinit();

    var wide = try bias.broadcastTo(&ctx, .{ .batch, .d }, .{ 2, 3 });
    defer wide.deinit();
    var copied = [_]f32{0} ** 6;
    try wide.copyTo(&copied);
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30, 10, 20, 30 }, &copied);
}
```

## 7.7 Pointwise and gated broadcasting (`src/tag_ops.zig`)

```zig
pub const PointwiseOp = enum { add, sub, mul, div, max, min };

pub fn pointwise(comptime op: PointwiseOp,
                 comptime left_tags: anytype, left: *const RawTensor, ctx: *ExecContext,
                 comptime right_tags: anytype, right: *const RawTensor) !RawTensor
pub fn gatedPointwise(comptime op: GatedOp, ...same signature...) !RawTensor
```

Broadcasting is entirely tag-driven. Any two tag sets are compatible at
comptime as long as their union fits `max_rank`; per-axis compatibility is a
runtime check. For each tag of `pointwiseResultTags(left_tags, right_tags)`
([§7.4](07-named-axes-the-tag-algebra.md#74-result-tag-computation-pointwise-and-dot-srctagszig)):

1. an operand missing the tag contributes dim 1;
2. equal dims pass through; a dim of 1 broadcasts to the other's dim
   (zero-stride, no copy); two unequal non-1 dims return
   `TensorError.ShapeMismatch`.

Both operands are broadcast to the result shape as views, then the
rank-matched ExecContext kernel runs (`add`/`sub`/`mul`/`div`/`max`/`min`
with an explicit `.f32`,
or `gated` for `gatedPointwise`). `GatedOp` ([§4](04-tensor-operations.md)) is
`enum { glu, swiglu, geglu, situ }`; every member computes
an `up`-side transform of `left` times a gate activation of `right`
(`gatedSourceScalar` × `gatedActivationScalar`, `src/backend/ops.zig` —
the source transform is the identity for the classic ops):
`left * σ(right)`, `left * silu(right)`, `left * gelu(right)`; the MoE
entries take an `exec.Gated{ .op, .clamp }` whose optional clamp bounds both
operands first (`clamp(left, ±c) * silu(min(right, c))`, DeepSeek V4's
clamped SwiGLU at `c = 10`; [§4.18](04-tensor-operations.md#418-moe-facade-entries-srcexecmoezig-srcexeczig)),
and `25·tanh(left/25) * 4·tanh(right/4)·σ(right)` for `.situ` (Kimi K3's
SiTU: a soft-bounded SiLU gate over a soft-clamped up input). The output
is always a newly materialized contiguous tensor in result-tag order.

The shape half is exposed separately for callers that need it (VJPs, facade):

```zig
pub fn pointwiseShape(comptime tensor_dtype: DType, comptime result_tags: anytype,
                      comptime left_tags: anytype, left: *const TensorOf(tensor_dtype),
                      comptime right_tags: anytype,
                      right: *const TensorOf(tensor_dtype)) ![rawRank(result_tags.len)]usize
```

These report the **raw-rank** shape — a scalar result reports `{1}` — and
perform the same dim-by-dim validation.

On the facade this is `add`/`sub`/`mul`/`div`/`maximum`/`minimum` and
`gated`/`glu`/`swiglu`/`geglu`/`situ`; the result type is
`Tensor(pointwiseResultTags(...))` ([§4](04-tensor-operations.md)).

```zig
test "pointwise broadcasts by tag" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

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

Disjoint tag sets broadcast to their union — an outer product without any
reshape ceremony:

```zig
test "disjoint tags broadcast to the union" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var rows = try fucina.Tensor(.{.batch}).fromSlice(&ctx, .{2}, &.{ 1, 10 });
    defer rows.deinit();
    var cols = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer cols.deinit();

    var outer = try rows.mul(&ctx, &cols); // result tags {.batch, .d}
    defer outer.deinit();
    const shape = outer.shape();
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, &shape);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 10, 20, 30 }, try outer.dataConst());
}
```

Same-tag axes with conflicting sizes fail at runtime:

```zig
test "incompatible dims fail with ShapeMismatch" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 10, 20 });
    defer b.deinit();

    try std.testing.expectError(error.ShapeMismatch, a.add(&ctx, &b));
    try std.testing.expectError(error.ShapeMismatch, a.dot(&ctx, &b, .d));
}
```

## 7.8 Split, merge, flatten, and multi-axis reduction (`src/tag_ops.zig`)

```zig
pub fn splitAxisView(comptime source_tags: anytype, source: *const RawTensor,
                     comptime tag: Tag, comptime split_tags: anytype,
                     split_shape: [split_tags.len]usize) !RawTensor
pub fn mergeAxesView(comptime source_tags: anytype, source: *const RawTensor,
                     comptime out_tag: Tag, comptime merge_tags: anytype) !RawTensor
pub fn flattenTensor(ctx: *ExecContext, source: *const RawTensor) !RawTensor
pub fn sumManyTensor(comptime tags: anytype, source: *const RawTensor,
                     ctx: *ExecContext, comptime reduce_tags: anytype) !RawTensor
```

**`splitAxisView`** factors one axis into several named factor axes, zero-copy
on **any** source layout: factor strides derive from the split axis's own
stride (`stride(axis) × suffix-product of remaining factors`), so strided
views split fine. The factor dims must be non-zero and multiply exactly to the
source axis dim (`TensorError.InvalidShape` otherwise). Tag-level constraints
come from `splitTags` ([§7.3](07-named-axes-the-tag-algebra.md#73-tuple-rewrites-and-axis-maps-srctagszig)).

**`mergeAxesView`** is the inverse: it collapses adjacent axes into one,
zero-copy — but only when the merged axes are **stride-compatible**, i.e. laid
out as an unsplit axis: for each adjacent pair,
`stride(i) == shape(i+1) × stride(i+1)`. A transposed or otherwise gapped
layout returns `TensorError.UnsupportedView` (no silent materialization; make
the tensor contiguous first, e.g. facade `materialize`, [§4](04-tensor-operations.md)). The merged axis
takes the stride of the last merged axis; the merged dim product is
overflow-checked. Tag-level contiguity (`mergeTags`, [§7.3](07-named-axes-the-tag-algebra.md#73-tuple-rewrites-and-axis-maps-srctagszig)) is checked at
comptime; stride compatibility is the runtime half of the same rule.

**`flattenTensor`** returns a rank-1 tensor of all elements in logical order:
a zero-copy reshape when the source is contiguous, otherwise it materializes
through the ExecContext first (owned result either way).

**`sumManyTensor`** reduces away `reduce_tags` (comptime: unique, subset of
`tags`). Fast paths: an empty reduce set returns a `cloneView`; reducing every
tag lowers to the full reduction `ctx.sum` (scalar `{1}`). Otherwise it walks
the reduce axes innermost-first via `reduceAxesDescending`
([§7.3](07-named-axes-the-tag-algebra.md#73-tuple-rewrites-and-axis-maps-srctagszig)) so remaining axis indices stay valid: a run of
adjacent reduce axes (the bias-gradient case `{batch, seq}` of
`[batch, seq, d]`) is one merged axis (a contiguous reshape; a non-contiguous
source is materialized first) and takes one `ctx.sumAxis` pass, every other
axis one pass each.

Facade equivalents ([§4](04-tensor-operations.md)): `split`, `merge` (both differentiable view ops),
`flatten(ctx, out_tag)` → `Tensor(.{out_tag})`, `sumMany` →
`Tensor(removeTags(tags, reduce_tags))`, and `sumAll` → `Tensor(.{})`.

```zig
test "split and merge rename factor axes as views" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

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

```zig
test "merge rejects stride-incompatible layouts" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .a, .b }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    var t = try x.permuteTo(&ctx, .{ .b, .a }); // zero-copy transposed view
    defer t.deinit();

    // Tag-contiguous, but the transposed strides cannot collapse into one axis.
    try std.testing.expectError(error.UnsupportedView, t.merge(&ctx, .m, .{ .b, .a }));
}
```

```zig
test "sumMany reduces several named axes at once" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .batch, .seq, .d }).fromSlice(&ctx, .{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    var d_totals = try x.sumMany(&ctx, .{ .batch, .seq }); // tags {.d}
    defer d_totals.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 22, 26, 30 }, try d_totals.dataConst());
}
```

## 7.9 `taggedDot`: tag-directed contraction and its lowering (`src/tag_ops.zig`)

```zig
pub fn taggedDot(comptime left_tags: anytype, left: *const RawTensor, ctx: *ExecContext,
                 comptime right_tags: anytype, right: *const RawTensor,
                 comptime contract_tag: Tag) !RawTensor
```

Semantics: contract the `contract_tag` axis of both operands; every shared
non-contracted tag is a **batch axis**; the result tags are
`dotResultTags` = batch ++ left free ++ right free ([§7.4](07-named-axes-the-tag-algebra.md#74-result-tag-computation-pointwise-and-dot-srctagszig)). Because the
contraction is named, the operands' physical axis order never changes the
mathematical result — it only selects the kernel.

Validation happens before any compute, via:

```zig
pub fn dotResultShape(comptime left_dtype: DType, comptime right_dtype: DType,
                        comptime left_tags: anytype, left: *const TensorOf(left_dtype),
                        comptime right_tags: anytype, right: *const TensorOf(right_dtype),
                        comptime contract_tag: Tag) ![rawRank(dotResultTags(...).len)]usize
```

which requires the contract dims to be equal and every batch tag's dim to
match on both sides (`TensorError.ShapeMismatch`), and returns the raw-rank
result shape (`{1}` for a scalar result). Dot batching is exact-match — batch
dims do **not** broadcast (unlike pointwise); use the facade `matmul` with
explicit `out_tags` for stride-0 broadcast batching ([§4](04-tensor-operations.md)).

Lowering: `taggedDot` is a one-line delegation to `taggedEinsum` with
`dotResultTags(left_tags, right_tags, contract_tag)` as the equation — see
the taggedEinsum subsection below for the full pipeline (role assignment,
runtime orientation selection, batch collapse). The classic dot dispatches
fall out of it: vector·vector runs `ctx.dot`; 2-D layouts pick
`matmul2D`/`matmulTransA`/`matmulTransB` by whichever aligned orientation is
contiguous (the `(0,1)` layout wants both transposes — only one is available
per call, so the smaller operand materializes); canonical batched layouts
hit `bmm`/`bmmTransA`/`bmmTransB` with no data movement; vectors ride along
as size-1 GEMM axes; everything else aligns and materializes at most once
per operand. Kernel outputs are contiguous; a final zero-copy reshape
restores the canonical per-axis result shape.

`taggedDot` itself is the f32 path. The facade `dot` ([§4](04-tensor-operations.md)) has the same tag
semantics for every RHS dtype but routes quantized-block, `f16`, and `bf16`
RHS tensors to dedicated kernels ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md), [§10](10-quantization.md)), and attaches `DotBackward` /
`ConstRhsDotBackward` for autograd ([§5](05-automatic-differentiation.md)). Its result type is
`Tensor(dotResultTags(tags, other_tags, contract_tag))`.

```zig
test "dot treats shared non-contracted tags as batch axes" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .batch, .m, .k }).fromSlice(&ctx, .{ 2, 2, 3 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .batch, .k, .n }).fromSlice(&ctx, .{ 2, 3, 2 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer b.deinit();

    var c = try a.dot(&ctx, &b, .k); // result tags {.batch, .m, .n}
    defer c.deinit();
    const shape = c.shape();
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2 }, &shape);
    try std.testing.expectEqualSlices(f32, &.{ 22, 28, 49, 64, 220, 244, 301, 334 }, try c.dataConst());
}
```

```zig
test "dot contracts by tag regardless of physical axis order" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .m, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .n, .k }).fromSlice(&ctx, .{ 2, 3 }, &.{ 7, 9, 11, 8, 10, 12 });
    defer b.deinit();

    var c = try a.dot(&ctx, &b, .k); // lowers to the trans-B matmul kernel
    defer c.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 58, 64, 139, 154 }, try c.dataConst());
}
```

```zig
test "contracting the only tag yields a scalar tensor" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer a.deinit();
    var b = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 4, 5, 6 });
    defer b.deinit();

    var y = try a.dot(&ctx, &b, .d); // Tensor(.{}): no tags, raw shape {1}
    defer y.deinit();
    comptime std.debug.assert(@TypeOf(y).tag_count == 0);
    try std.testing.expectEqual(@as(f32, 32), try y.item());
}
```

### `taggedEinsum`: multi-index contraction lowering

```zig
pub fn taggedEinsum(comptime left_tags: anytype, left: *const RawTensor, ctx: *ExecContext,
                    comptime right_tags: anytype, right: *const RawTensor,
                    comptime out_tags: anytype) !RawTensor
pub fn einsumResultShape(comptime left_dtype: DType, comptime right_dtype: DType,
                           comptime left_tags: anytype, left: *const TensorOf(left_dtype),
                           comptime right_tags: anytype, right: *const TensorOf(right_dtype),
                           comptime out_tags: anytype) ![rawRank(out_tags.len)]usize
```

`taggedEinsum` is the raw lowering behind the facade `einsum` AND `dot`
([§4.8](04-tensor-operations.md#48-dot-tag-directed-contraction-srcagtensorzig-srctag_opszig)) — `taggedDot` delegates here with the canonical dot result order as
the equation. The output tag tuple is the whole equation, axis roles come
from `einsumPartTags` ([§7.4](07-named-axes-the-tag-algebra.md#74-result-tag-computation-pointwise-and-dot-srctagszig)), and every shared dim is validated by
`einsumResultShape` before compute. The lowering, in order:

1. **Pre-sum** — operand-private dropped tags are reduced away with
   `sumManyTensor`, so every remaining axis is batch/free/contract.
2. **Scalar output** — flatten both operands (right aligned to the left's
   axis order) and run `ctx.dot`.
3. **Role assignment** (comptime) — when `out_tags` nests as
   `[batch][left free][right free]` the operands keep their roles; the
   swapped nesting `[batch][right free][left free]` swaps kernel-left and
   kernel-right (so "double-transposed" layouts run as one plain GEMM); an
   interleaved `out_tags` contracts in canonical order and pays one output
   materialization (`permuteTensorTo` + materialize) at the end.
4. **Orientation selection** (runtime) — both operands are aligned to the
   group-nested order as zero-copy permute views, and each side
   independently picks the plain or transposed kernel layout by probing
   which aligned view is already contiguous (a trans GEMM is free; a
   materialize costs a copy pass). At most one of transA/transB
   can be taken per call — when both operands prefer transposed, the larger
   keeps it and the smaller is materialized. Groups then collapse by
   zero-copy reshape into `[batch…,m,k]·[batch…,k,n]` (or the trans
   permutations) and one `matmul2D`/`matmulTransA`/`matmulTransB` (no
   batch) or `bmm`/`bmmTransA`/`bmmTransB` runs; a side whose no
   orientation is contiguous materializes once.

The batch group collapses into a single bmm batch axis before the kernel
call, so any batch count the operands can represent is lowerable (there is
no rank-(batch+2) cap). The facade attaches `EinsumBackward` ([§5.8](05-automatic-differentiation.md#58-vjp-coverage-inventory-srcagbackward)), whose two
branches are einsums themselves — the gradient of a contraction is a
contraction, so no pointwise fallback exists anywhere on the contraction
backward paths (`DotBackward` and `ConstRhsDotBackward` delegate to the
einsum records).

## 7.10 Shared dtype-generic helpers (`src/tag_ops.zig`)

```zig
pub fn contiguousForReshape(comptime tensor_dtype: DType, ctx: *ExecContext,
                            value: *const TensorOf(tensor_dtype)) !TensorOf(tensor_dtype)
pub fn productRange(comptime tensor_dtype: DType, value: *const TensorOf(tensor_dtype),
                    comptime start: usize, comptime count: usize) usize
```

`contiguousForReshape` returns a `cloneView` when the value is already
contiguous, otherwise a materialized copy via the ExecContext — an owned
tensor either way, so callers can `defer deinit` unconditionally.
`productRange` multiplies a comptime-bounded run of dims; the dot paths use
it to collapse free/batch axis groups. Both are used by the facade's typed dot
paths (`ag/tensor.zig`) as well as the library itself.
