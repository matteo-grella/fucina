# 4. Tensor operations

This section is the reference for the math/NN operation surface of the public
autograd `fucina.Tensor(tags_spec)` facade (`src/ag/tensor.zig`), the
tag-semantics lowering library behind it (`src/tag_ops.zig`), and the public
option types those operations take (`src/exec.zig`). Construction, data
access, and structural views (`withTags`, `permuteTo`, `transpose`,
`alignTo`, `insertAxis`, `squeeze`, `split`, `merge`, `broadcastTo`,
`viewWithStrides`, `flatten`, `materialize`, `detach`) are covered in [§3](03-tensors-types-construction-and-data-access.md); the
tag algebra itself in [§7](07-named-axes-the-tag-algebra.md); the autograd engine driving the backward records
named here in [§5](05-automatic-differentiation.md).

## 4.1 The common operation contract (`src/ag/tensor.zig`)

Every operation below shares one contract, implemented by the shared tails
`finishOp`/`finishNoGrad`:

- **Signature shape.** Ops are methods on `Tensor(tags)` taking
  `ctx: *ExecContext` as the first runtime argument. Axes are chosen by
  comptime tag (`comptime tag: Tag`); misnaming a tag that the tensor does
  not carry is a **compile error**, never a runtime error. Shape problems the
  type system cannot see (mismatched dims, bad lengths) are recoverable
  `TensorError`s (`ShapeMismatch`, `InvalidShape`, `InvalidDataLength`,
  `IndexOutOfBounds`, integer division's `DivisionByZero`); a non-shape
  argument failing its own validity check (a dropout `p` outside `[0, 1)`,
  a non-positive `softcap` cap, `clamp` with `min > max`, a Huber `delta`
  that is not positive and finite, a negative standardize `eps`, `topk`
  with `k == 0`, an ALiBi/sinks softmax option combination naming no
  target) is `InvalidArgument`; the data-dependent no-match outcome of
  `maskedSelect`/`maskedScatter` gets the dedicated `EmptySelection` so it
  stays catchable apart from those. This is the whole recoverable
  vocabulary of the op surface: `fucina.Error` names the merge
  (`fucina.TensorError`, the graph-control names `UnsupportedGradient`,
  `MutableDataRequiresNoGrad`, `NoGradientGraph`,
  `ActiveExecScopeUnsupported`, the backward engine's
  `MissingOutputGradient`/`MissingBackwardGradient`/`BackwardAlreadyRun`,
  and `OutOfMemory`) for wrappers that thread any fucina error upward;
  each method still exposes its precise inferred error set.
- **Ownership.** Each op allocates and returns a **new owned tensor**; the
  caller `deinit`s it. Operands are borrowed via `*const` and never consumed
  (the two `take*` ops in [§4.3](04-tensor-operations.md#43-scalar-variants-and-in-placeno-grad-helpers-srcagtensorzig) are the documented exception). While an exec
  scope is open on the context, returned tensors are scope-owned borrows and
  their `deinit` is a safe no-op ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)).
- **Gradients.** A backward record is attached iff at least one operand
  `requiresGrad()` and gradients are globally enabled (`fucina.noGrad`, [§5](05-automatic-differentiation.md)).
  Families that are no-grad by design, or that restrict which operands
  receive gradients, say so inline; grad-incompatible calls fail with
  `error.UnsupportedGradient` (or a more specific error named per family).
  Ops **composed** from other facade ops (`nllLoss`, `l2Normalize`,
  `cosineSimilarity`, `maskedSelect`, `stack`, `unbindInto`) release their
  intermediates on return; the consumer records retain the graph nodes, so
  they differentiate with or without an exec scope ([§5](05-automatic-differentiation.md)).
- **Thread-safety.** A context is single-threaded at the API surface: run
  ops on one `ExecContext` from one thread ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)). Kernels parallelize
  internally through the context's work pool ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)).
- **Name suffixes.** Three suffixes mark departures from the ownership
  contract above, spelled in the name at every site: `*InPlace` mutates
  `self`'s storage and returns nothing new; `take*` consumes a unique
  (non-view) input tensor, which must not be used afterwards; `*Into`
  writes into a caller-provided output instead of allocating. Everything
  unsuffixed allocates and returns a new owned tensor.
- **Option types.** Options are passed as literals, so their type names are
  rarely written out. The ones re-exported at the `fucina` root are
  `UnaryOp`, `Reduction`, `CrossEntropyOptions`, `StandardizeOptions`,
  `StandardizeAccumulation`, `StandardizeEpsMode`, `RouterTopKOptions`,
  `RopeMode`, `RopeTable`, `RopeTheta`, `MoeRhs`, `MoeBatchProfile`,
  `PackedRhs`, `GatedOp`, `VarianceOptions`. The remaining option types
  named in this section (`CompareOp`, `MatmulKind`, `MseOptions`,
  `HuberOptions`, `BceOptions`, `KlDivOptions`, `LinearDistillOptions`,
  `SoftmaxExtOptions`, `NormOrder`) live in
  `src/exec.zig` and are reached through enum/struct literals at call sites
  (`.swiglu`, `.lt`, `.trans_b`, `.{ .reduction = .none }`).
  A few option sets (`softmax`, `layerNorm`, the masked `sum`/`mean`/
  `max`/`min`, `standardizeAxis`, `linearRecurrence`, `groupedAttention`)
  are deliberately `anytype` rather than one struct type: they carry
  caller-typed tensor operands (a mask, affine weights, an initial state,
  each keeping its own tensor type and tags) or comptime tag fields, which
  a single runtime struct cannot hold. They validate their fields at
  comptime the same way (a misspelled field is a compile error naming the
  op and the field).

Snippets in this section are runnable test blocks and assume:

```zig
const std = @import("std");
const fucina = @import("fucina");
```

## 4.2 Pointwise binary ops and tag-driven broadcasting (`src/ag/tensor.zig`, `src/tag_ops.zig`)

```zig
pub fn add(self: *const Self, ctx: *ExecContext, other: anytype)
    !Tensor(pointwiseResultTags(tags, TensorObject(@TypeOf(other)).axis_tags))
// same shape for: sub, mul, div; TensorObject unwraps pointer operands
```

`add`, `sub`, `mul`, `div` broadcast **by tag name**, not by position. The
result tag set is `pointwiseResultTags`: `self`'s tags in order, followed by
`other`'s tags that `self` does not carry. Per shared tag the dims must be
equal or one of them 1; a tag missing from one operand behaves as dim 1.
Broadcasting is a zero-stride view (no materialization); when both operands
have identical tags and shapes the kernel runs directly with no view step.
`other` may be a tensor value or pointer. Backward: full two-operand VJP; a
gradient flowing into a broadcast operand is reduced back over the broadcast
axes ([§5](05-automatic-differentiation.md)).

Three more binary pointwise ops share the same tag-broadcast rule and
two-operand VJP:

- `maximum(ctx, other)` / `minimum(ctx, other)` — torch.maximum/minimum,
  full `.max`/`.min` members of the binary kernel enum (pooled SIMD, the
  `add`/`mul` tier): NaN in either operand propagates NaN (NOT the IEEE
  maxNum rule bare `@max` follows); the gradient goes to the winning
  operand and is split evenly on exact ties (torch's subgradient, ±inf
  ties included).
- `pow(ctx, other)` — `self ^ other` with `std.math.pow` domain semantics
  (negative base + non-integer exponent is NaN, `0^0 = 1`); `powScalar`
  ([§4.3](04-tensor-operations.md#43-scalar-variants-and-in-placeno-grad-helpers-srcagtensorzig)) is the scalar-exponent fast path. Implemented over the elemental
  tier ([§4.4](04-tensor-operations.md#44-unary-ops-srcagtensorzig-srcbackendopszig)) — `std.math.pow` has no portable SIMD form with these
  domain semantics. The exponent-side gradient `ln(a)·a^b` is meaningful
  only for positive bases, as in torch.

```zig
test "pointwise add broadcasts by tag" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var b = try fucina.Tensor(.{.col}).variableFromSlice(&ctx, .{2}, &.{ 10, 20 });
    defer b.deinit();

    var y = try x.add(&ctx, &b); // result tags .{ .row, .col }; b broadcasts over .row
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 13, 24 }, try y.dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gb = (try b.grad(&ctx)).?; // broadcast VJP: gradient reduced back to .{ .col }
    defer gb.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 2 }, try gb.dataConst());
}
```

## 4.3 Scalar variants and in-place/no-grad helpers (`src/ag/tensor.zig`)

Scalar variants (all differentiable, all return a new tensor):

| Method | Value | Backward |
|---|---|---|
| `scale(ctx, s)` | `x·s` | `ScaleBackward` |
| `addScalar(ctx, s)` | `x + s` | pass-through |
| `subScalar(ctx, s)` | `addScalar(-s)` | pass-through |
| `divScalar(ctx, s)` | `scale(1/s)` | via `scale` |
| `powScalar(ctx, e)` | `x^e` (positive `x`) | `e·x^(e−1)` |

```zig
test "scalar op variants" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 3, 4 });
    defer x.deinit();
    var shifted = try x.addScalar(&ctx, 1); // {4, 5}
    defer shifted.deinit();
    var squared = try shifted.powScalar(&ctx, 2); // {16, 25}
    defer squared.deinit();
    var y = try squared.scale(&ctx, 0.5); // {8, 12.5}
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 8, 12.5 }, try y.dataConst());
}
```

Inference-oriented helpers (the in-place and consuming ones reject
grad-requiring operands with `error.UnsupportedGradient`; `biasAdd` does
not — see its bullet):

- `addAxisVectorInPlace(ctx, bias, axis_tag)` — adds a `[axis_dim]` f32 row
  vector along the **last** axis `axis_tag`, mutating `self` in place.
- `addAxisVectorUnaryInPlace(ctx, op, bias, axis_tag)` — fused bias-add +
  `UnaryOp` activation, in place.
- `addScaledInPlace(ctx, other, alpha)` — `self += alpha·other` (same
  shape), in place.
- `takeAddNoGrad(ctx, other)` / `takeScaleNoGrad(ctx, s)` — **consume**
  `self` (it becomes `undefined`) and return the result, reusing `self`'s
  storage when the runtime can take it in place. They additionally fail with
  `error.ActiveExecScopeUnsupported` on a scope-owned borrow.
- `biasAdd(ctx, bias, axis_tag)` — the out-of-place variant: a new tensor,
  `self` unchanged. It accepts grad-requiring input: `self`'s gradient
  passes through identity; the raw `bias` slice receives none. For a
  trainable bias, use broadcast `add`.

## 4.4 Unary ops (`src/ag/tensor.zig`, `src/backend/ops.zig`)

```zig
pub fn unary(self: *const Self, ctx: *ExecContext, comptime op: UnaryOp) !Self
```

`exec.UnaryOp` is the closed kernel enum; most values also have a direct
method alias:

| `UnaryOp` | Method | Notes |
|---|---|---|
| `.relu` | `relu` | dedicated backward record |
| `.exp` | `exp` | |
| `.sqrt` | `sqrt` | |
| `.rsqrt` | `rsqrt` | |
| `.sigmoid` | `sigmoid` | |
| `.silu` | `silu` | |
| `.log` | `log` | |
| `.log1p` | `log1p` | `log(1 + x)` |
| `.softplus` | — (`unary(.softplus)` only) | `log(1 + e^x)`, sign-stable (torch softplus, pre-threshold regime) |
| `.neg` | `neg` | |
| `.abs` | `abs` | |
| `.sin` | `sin` | |
| `.cos` | `cos` | |
| `.tanh` | `tanh` | |
| `.fast_tanh` | `fastTanh` | NAM rational approximation |
| `.gelu` | `gelu` | tanh approximation (exact form of `.gelu_quant`) |
| `.quick_gelu` | `quickGelu` | `x·sigmoid(1.702x)` |
| `.gelu_quant` | — (`unary(.gelu_quant)` only) | ggml GGML_GELU_FP16 parity: f16-rounded tanh-gelu with hard clamps |
| `.elu` | `elu` | alpha = 1, matches `ggml_vec_elu_f32` |
| `.gelu_erf` | `geluErf` | exact-erf GELU (musl `erff` translation, matches `ggml_vec_gelu_erf_f32`) |
| `.erf` | `erf` | error function (torch.erf); the same musl-faithful `erff` as `.gelu_erf` |
| `.floor` | `floor` | zero gradient a.e. (torch convention) |
| `.ceil` | `ceil` | zero gradient a.e. |
| `.round` | `round` | round-half-to-EVEN (torch.round), NOT half-away; 2^23 magic-number trick, scalar and SIMD legs bit-identical; zero gradient a.e. |
| `.sign` | `sign` | ±0 preserved, NaN propagates (numpy/torch); zero gradient a.e. |
| `.reciprocal` | `reciprocal` | `1/x`; output-derivative backward (`-out²`, like tanh) |

All unary ops are differentiable ([§5](05-automatic-differentiation.md)). Related parameterized elementwise
ops:

- `leakyRelu(ctx, negative_slope)` — differentiable, dedicated backward.
- `clamp(ctx, min_value, max_value)` — differentiable; gradient is zero
  outside `[min, max]`.
- `clampMin(ctx, min_value)` / `clampMax(ctx, max_value)` — one-sided
  clamps (torch's `clamp_min`/`clamp_max`); the open side is ±inf, so
  values and gradients pass through it untouched.
- `dropout(ctx, p, seed)` — inverted dropout: keeps `x[i]/(1−p)` iff the
  per-element counter RNG at `(seed, i)` draws below `1−p`. The mask is
  never stored: forward, backward, and checkpoint recompute regenerate it
  from `(seed, index)`, so the op is a pure function of `(input, p, seed)`.
  Requires `0 <= p < 1`; `p == 0` returns an identity view. Pass a fresh
  seed per call (reusing a seed reuses the mask); eval mode is caller-side —
  do not call dropout at eval.

```zig
test "unary ops" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ -1, 0, 1 });
    defer x.deinit();
    var r = try x.relu(&ctx);
    defer r.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1 }, try r.dataConst());
    var s = try x.silu(&ctx); // same as x.unary(&ctx, .silu)
    defer s.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), (try s.dataConst())[2], 1e-6);
}
```

### Elemental ops: user-defined scalar functions (`src/ag/elemental.zig`)

```zig
pub fn elementalUnary(self, ctx, comptime Op: type, extra: anytype) !Self
pub fn elementalBinary(self, ctx, other: anytype, comptime Op: type, extra: anytype)
    !Tensor(pointwiseResultTags(...))  // the standard pointwise tag rule
```

`UnaryOp` is a closed enum; `elementalUnary`/`elementalBinary` are the
user-extensible escape hatch — a convenience tier over `customVjp` ([§5.6](05-automatic-differentiation.md#56-custom-vjps-srcagcustomzig))
that lifts a comptime scalar `Op` to a differentiable eager tensor op. The
user writes scalar math only; the adapter owns buffer plumbing,
strided-input materialization, tag-driven broadcasting, broadcast-gradient
sum-reduction, `needs_grad` pruning, and the worker-team chunking of the
scalar loops (bitwise thread-count-neutral: disjoint pure writes).

<!-- snippet: helper -->
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

Binary `Op` declares `forward(a, b, extra)` plus `backwardA`/`backwardB`
returning dL/da and dL/db evaluated elementwise at the broadcast result
shape; broadcast operands get their gradient sum-reduced back to their own
tags/shape exactly like `add`/`mul`. Missing declarations are compile
errors. Both ops are f32-branch only, accept strided views, and return
owned contiguous results; `extra` is captured **by value** in the backward
node (the `customVjp` lifetime contract: pointees must outlive backward).
Validate a new `Op` with `fucina.gradcheck` ([§5.7](05-automatic-differentiation.md#57-gradient-checking-srcaggradcheckzig)).

An `Op` may additionally declare vector twins over `fucina.simd.Vf32`
lanes — `forwardVec`/`backwardVec` (unary), `forwardVec` plus
`backwardAVec`/`backwardBVec` (binary) — and the kernels then run those on
full vectors with the scalar rules on the tail (the built-in kernels'
lane/tail split). Bodies containing transcendentals want this: a scalar
rule compiles to one libm call per element, where a vector body can use
the `fucina.simd` helpers (`vexpf`, `sigmoidVec`, `tanhVec`; `vector_len`
is the lane count — [§9.4](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#94-native-backend-portable-vector-kernels-srcbackendvector)). The scalar and vector rules must compute the
same function; lane/tail rounding may differ exactly as it does in the
built-ins.

```zig
test "elementalUnary" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).variableFromSlice(&ctx, .{3}, &.{ 1, -2, 3 });
    defer x.deinit();
    var y = try x.elementalUnary(&ctx, Square, {});
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 4, 9 }, try y.dataConst());

    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, -4, 6 }, try gx.dataConst());
}
```

## 4.5 Gated activations (`src/ag/tensor.zig`)

```zig
pub fn gated(self, ctx, other: anytype, comptime op: GatedOp) !Tensor(...)  // pointwise result tags
pub fn splitGated(self, ctx, comptime op: GatedOp, comptime tag: Tag, comptime out_tag: Tag)
    !Tensor(replaceTag(tags, tag, out_tag))
```

`exec.GatedOp` is `{ .glu, .swiglu, .geglu, .situ }`. The
two-operand form computes `self * act(other)` — the **second** operand is
the gate — with the same tag-broadcast rule as [§4.2](04-tensor-operations.md#42-pointwise-binary-ops-and-tag-driven-broadcasting-srcagtensorzig-srctag_opszig): `glu` =
`self·sigmoid(other)`, `swiglu` = `self·silu(other)`, `geglu` =
`self·gelu(other)` (tanh approximation; Gemma's GeGLU). `.situ` (Kimi
K3's SiTU) is the one member that also transforms the up side:
`25·tanh(self/25) · 4·tanh(other/4)·sigmoid(other)` — a soft-bounded SiLU
gate (beta 4) on a soft-clamped up input (linear beta 25).
`glu`/`swiglu`/`geglu`/`situ` are direct aliases of `gated(..., op)`.
Differentiable in both operands. The MoE entries ([§4.18](04-tensor-operations.md#418-moe-facade-entries-srcexecmoezig-srcexeczig)) take the
activation with its parameter, `exec.Gated{ .op, .clamp }`: with `clamp`
set, the gate is `min(gate, clamp)` before the activation and `up` is
clamped to `[-clamp, clamp]` (DeepSeek V4's clamped SwiGLU is
`.{ .op = .swiglu, .clamp = 10 }`); `gated` and `splitGated` take the
function only.

`splitGated` halves axis `tag` and gates one half with the other in a single
fused kernel; the gate-half conventions differ deliberately (ggml parity):
`.swiglu` gates with the **first** half (`silu(first)·second`), `.glu` with
the **second** (`first·sigmoid(second)`). `out_tag == tag` is allowed.
`.geglu` and `.situ` are compile errors (no split kernel exists for
either; K3 projects gate and up separately — use the pointwise `situ`).
Differentiable in `self`.

```zig
test "gated pointwise and split-gated" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var up = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{1}, &.{2});
    defer up.deinit();
    var gate = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{1}, &.{1});
    defer gate.deinit();
    var y = try up.swiglu(&ctx, &gate); // up * silu(gate)
    defer y.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2 * 0.7310586), (try y.dataConst())[0], 1e-6);

    var fused = try fucina.Tensor(.{.ff}).fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer fused.deinit();
    var z = try fused.splitGated(&ctx, .swiglu, .ff, .d); // silu(first half) * second half
    defer z.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586 * 2), (try z.dataConst())[0], 1e-6);
}
```

## 4.6 Masks, comparisons, and conditionals (`src/ag/tensor.zig`)

Mask producers emit `.bool` tensors (torch's comparison dtype); mask
consumers (`where`, `maskedFill`, the logical ops) take a `.bool` mask or
a float tensor read by truthiness (`!= 0`; NaN truthy). Like every typed
constant, `.bool` results are CALLER-owned even under an exec scope. Count
a mask with `sum`/`sumAll` (i64, [§4.19](04-tensor-operations.md#419-math-on-non-f32-tensors-srcagtensorzig)) and cast with `to(.f32)` for the
mask-multiply idiom.

- `compare(ctx, op, other)` — `.bool`, true where `self <op> other`
  holds. `op` is `exec.CompareOp` (`.eq .ne .lt .le .gt .ge`); `other` is
  comptime-dispatched: a same-tagged tensor (same shape only) or a
  numeric scalar. **No-grad by design** (constant mask). NaN follows
  IEEE: any comparison involving NaN is false except `.ne`, which is
  true. Also on the typed branches: f16/bf16 compare through f32; INTEGER
  tensors compare natively (exact at any magnitude — token-id masks).
- `logicalAnd`, `logicalOr`, `logicalXor` (`ctx, other`) and
  `logicalNot(ctx)` — elementwise logic over truthiness, `.bool` out.
  Defined on the f32 branch (float `self`, `.bool`-or-float `other`) and
  on the `.bool` branch itself (mask combinators); same shape only.
- `where(ctx, cond, other)` — `cond[i] ? self[i] : other[i]`.
  Differentiable in `self` and `other`; `cond` (`.bool` or float) is a
  non-grad mask.
- `maskedFill(ctx, mask, value)` — `mask[i] ? value : self[i]`.
  Differentiable in `self` (gradient zeroed where filled); `value` is a
  constant.
- `isnan(ctx)` / `isinf(ctx)` / `isfinite(ctx)` — torch's float
  predicates as `.bool` masks, built purely from the IEEE `compare`
  semantics (`isnan` is the self-`.ne` test; `isfinite` is
  `-inf < x < inf`, false for NaN and both infinities).
  Non-differentiable, unscoped-safe.
- `any(ctx, tag)` / `all(ctx, tag)` — `.bool`, true where any/every
  element along `tag` is truthy (NaN is truthy, the torch.any/all
  convention), the tag removed; `anyAll(ctx)`/`allAll(ctx)` are the
  scalar full-tensor forms (torch with no dim). Non-differentiable
  (compare → i64 count → compare), unscoped-safe.
- `bandMask(ctx, raw_shape, lower, upper)` — the banded / sliding-window
  attention-mask CONSTRUCTOR, on two-tag `.bool` types only: element
  `(i, j)` is true iff `i - j <= lower` and `j - i <= upper`, a null
  bound unbounded on that side (bounds are signed `?i64`). Causal
  keep-set = `(null, 0)`; sliding window of width W = `(W - 1, 0)`;
  tril(k) keep-set = `(null, k)`; triu(k) = `(-k, null)`. Feed it to
  `where`/`maskedFill` (via `broadcastTo` for batched scores) or cast
  with `to(.f32)` for the mask-multiply idiom; contradictory bounds
  yield an all-false mask, not an error.
- `bandPart(ctx, row_tag, col_tag, lower, upper)` — keep the `bandMask`
  band of the (`row_tag`, `col_tag`) plane and ZERO everything outside
  (tf.linalg.band_part with signed nullable bounds), at any rank
  carrying both tags — remaining axes pass through. Implemented as a
  constant 0/1 band-plane multiply, so it is differentiable in `self`
  with the exact same-band mask as VJP. `tril(ctx, row_tag, col_tag, offset)`
  / `triu(ctx, row_tag, col_tag, offset)` are the triangular special
  cases (torch.tril/triu with a diagonal offset): `bandPart(null, offset)`
  and `bandPart(-offset, null)` respectively.

```zig
test "bandMask bandPart tril build banded and triangular structure" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // Causal .bool keep-set for attention scores.
    const B = fucina.Tensor(.{ .dtype = .bool, .tags = .{ .q, .k } });
    var causal = try B.bandMask(&ctx, .{ 3, 3 }, null, 0);
    defer causal.deinit();
    try std.testing.expectEqualSlices(bool, &.{ true, false, false, true, true, false, true, true, true }, try causal.dataConst());

    // tril zeroes above the diagonal; the gradient is the same triangle.
    var x = try fucina.Tensor(.{ .row, .col }).variableFromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var lo = try x.tril(&ctx, .row, .col, 0);
    defer lo.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 3, 4 }, try lo.dataConst());
    var loss = try lo.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 1, 1 }, try gx.dataConst());
}

test "compare produces bool masks for maskedFill and where" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ -1, 0, 2 });
    defer x.deinit();
    var neg = try x.compare(&ctx, .lt, 0); // .bool: {true, false, false}
    defer neg.deinit();
    comptime std.debug.assert(@TypeOf(neg).dtype == .bool);
    var y = try x.maskedFill(&ctx, &neg, 0); // relu by hand
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 2 }, try y.dataConst());
    var n_neg = try neg.sumAll(&ctx); // count the mask: i64
    defer n_neg.deinit();
    try std.testing.expectEqual(@as(i64, 1), try n_neg.item());
}
```

## 4.7 Reductions and scans (`src/ag/tensor.zig`)

Axis reductions remove the reduced tag from the result type; `sumAll`
returns the scalar `Tensor(.{})`.

- `sum(ctx, tag)` / `mean(ctx, tag)` — differentiable.
- `sumMany(ctx, reduce_tags_spec)` — sums away several tags (innermost
  first); result tags `removeTags(tags, reduce_tags)`.
- `sumAll(ctx)` — full reduction to `Tensor(.{})`; read with `item()`.
- `variance(ctx, tag, options)` — `options: VarianceOptions`:
  `.{ .ddof = 0 }` = biased (LayerNorm convention), `.{}` (ddof 1) =
  Bessel-corrected (`torch.var` default). Differentiable.
- `cumsum(ctx, tag)` — inclusive prefix sum, shape-preserving;
  differentiable (gradient = reversed suffix sum). Both passes are serial
  per row by default, so results are bitwise deterministic for any thread
  count AND sequence-exact; `-Dvector-scan` ([§2.2](02-toolchain-build-and-project-wiring.md#22-build-options-buildzig)) vectorizes both
  (non-last axes stay bitwise identical; the last axis reassociates like
  `sum`'s SIMD lanes).
- `max(ctx, tag)` / `min(ctx, tag)` — extremum values with the tag removed
  (indices come from `argmax`/`topK`, [§4.16](04-tensor-operations.md#416-selection-argmax-topk-sort-routertopk-srcagtensorzig-srcexectopkzig)). The gradient flows only to the
  **first** occurrence of the extremum along the axis (strict-comparison
  tie-break, matching `torch.max` over a dim).
- `prod(ctx, tag)` — product with the tag removed (torch.prod over a
  dim), at `sum`'s kernel tier: rank-1 reduces through the pooled SIMD
  `prodInto`, a last-axis reduction runs one vectorized `prodSlice` per
  row (like `sum`, the SIMD lane order fixes the multiplication order per
  backend). Differentiable with torch's zero-handling: zero-free rows get
  `g·(Πx)/x_i`, exactly one zero routes the whole gradient to the zero
  slot, two or more kill the row's gradient.
- `cumprod(ctx, tag)` — inclusive running product, shape-preserving
  (torch.cumprod); serial per row by default, vectorized under
  `-Dvector-scan` ([§2.2](02-toolchain-build-and-project-wiring.md#22-build-options-buildzig), the `cumsum` gating). Differentiable: zero-free rows use the
  O(n) reverse-scan closed form; rows containing a zero fall back to an
  exact division-free O(n²) expansion (torch semantics).
- `segmentSum(ctx, tag, offsets)` — sums CONTIGUOUS index ranges of one
  axis (`torch.segment_reduce("sum")` over sorted segments):
  `out[..., i, ...] = Σ_{j ∈ [offsets[i], offsets[i+1])} x[..., j, ...]`.
  `offsets` must be non-decreasing, start at `0`, and end at the axis size
  (`offsets.len - 1` segments partitioning the whole axis; empty segments
  produce zero rows). The reduced axis keeps its tag at the new size
  `offsets.len - 1`. Differentiable: the gradient broadcasts each output
  row back over its segment. Rank-level entries on `ExecContext`:
  `segmentSum(rank, x, axis, offsets)` and its VJP helper
  `segmentBroadcast(rank, gy, axis, offsets, n)`.
- `linearRecurrence(ctx, time_tag, decay, options)` — the first-order
  linear recurrence `h_t = a_t ⊙ h_{t-1} + b_t` along `time_tag`,
  shape-preserving: `self` supplies `b` (and the result shape), `decay`
  supplies `a`, and every lane (each fixed choice of the non-time axes)
  scans independently — the associative-scan primitive of the
  SSM / linear-attention family (Mamba-2 / GDN-style diagonal decay
  states). `decay` broadcasts BY TAG like the pointwise ops (tags a
  subset of `self`'s; a zero-stride view read in place, never
  materialized): no time tag = static per-lane decay, `[time]` only =
  shared schedule, state-shaped tags = per-state decay.
  `options` is `.{}` or `.{ .initial = &h0 }` with `h0` carrying
  `self`'s tags minus `time_tag` (exact shape) — `h_{-1}` per lane, the
  streaming seam: feeding the previous chunk's last step continues a
  sequence bitwise-exactly. Determinism follows the `cumsum` contract:
  serial per lane, multiply-then-add per step; `-Dvector-scan`
  vectorizes non-last time axes across lanes BITWISE identically, and
  the last axis always stays serial (an in-register form would
  reassociate the recurrence). Differentiable in `self`, `decay`, and
  `initial`: the VJP is one reverse scan
  (`gh_t = a_{t+1}·gh_{t+1} + gy_t`; `decay` gets `gh_t·h_{t-1}`
  reduced back over its broadcast axes, `initial` gets `a_0·gh_0`).
  With `decay == 1` it degenerates to `cumsum` bitwise.
- `norm(ctx, tag, order)` / `normAll(ctx, order)` — vector norm along
  `tag` / over all elements (torch.linalg.vector_norm), `order` in
  `exec.NormOrder`: `.l1` = Σ|x|, `.l2` = sqrt(Σx²), `.inf` = max|x|.
  Composed from existing differentiable ops (scope-required under
  gradients); like torch, the `.l2` gradient at an all-zero vector is NaN
  (`sqrt'(0)`).

### Masked reductions

```zig
pub fn sum(self, ctx, comptime tag: Tag, opts: anytype) !Tensor(removeTag(tags, tag))
pub fn mean(self, ctx, comptime tag: Tag, opts: anytype) !Tensor(removeTag(tags, tag))
pub fn max(self, ctx, comptime tag: Tag, opts: anytype) !Tensor(removeTag(tags, tag))
pub fn min(self, ctx, comptime tag: Tag, opts: anytype) !Tensor(removeTag(tags, tag))
```

The `sum(a, dim, mask)` / `maxval(a, dim, mask)` family: a reduction
restricted to the elements a mask selects. `opts` is `.{}` (identical to the
unmasked op), `.{ .mask = &m }`, or `.{ .mask = &m, .empty = v }`; any other
field is a **compile error**, never a silently-unmasked reduction (the
`groupedAttention` opts discipline, [§4.13](04-tensor-operations.md#413-attention-srcagtensorzig)).

- **Mask contract**, shared with `where`/`maskedFill` ([§4.6](04-tensor-operations.md#46-masks-comparisons-and-conditionals-srcagtensorzig)): a `.bool` mask
  or a float read by truthiness (`!= 0`, NaN truthy), carrying `self`'s exact
  tags and shape, non-grad, integer masks a compile error. Compose
  `broadcastTo` for a mask over fewer axes.
- **Why not just compose.** The spelling these replace is `maskedFill` into a
  full-size f32 temporary followed by an ordinary reduction: two passes and a
  whole extra tensor. These accumulate straight out of the source through the
  same SIMD row kernel. Measured on M1 Max, ReleaseFast, `-Dblas=accelerate`
  (`zig build bench-masked-reduce`): **2.3–2.4x faster** than the composition
  at `[1024, 4096]` and `[4096, 1024]` on both the last-axis and
  non-last-axis arms.
- **Bitwise contract.** An excluded element contributes the operation's
  identity rather than being skipped, so a masked reduction performs exactly
  the composition's arithmetic in exactly its order: `sum` is **bitwise
  equal** to `maskedFill(¬mask, 0)` then `sum`, and bitwise equal to the
  unmasked reduction when the mask is all-true. Substituting the identity is
  also what keeps an excluded NaN from poisoning its lane.
- **Empty lanes.** A lane whose mask selects nothing takes the operation's
  identity: `0` for `sum`, `-inf`/`+inf` for `max`/`min` (Fortran's
  `maxval` of an empty selection is `-HUGE`). This is why a masked reduction
  needs no `EmptySelection` error the way `maskedSelect` does ([§3.7](03-tensors-types-construction-and-data-access.md#37-views-and-structural-ops-srcagtensorzig-srctag_opszig-srcexecgather_scatterzig)) — the
  answer is defined. `mean` has no identity, so its empty lane is `NaN`
  (0/0). `.empty = v` overrides the value for any of them.
- **Gradients.** `sum`: the unmasked scatter, zeroed where the mask
  excluded the element. `mean`: divided by the count of SELECTED elements
  per lane (not the axis length), so an empty lane — which produced a
  constant, not a function of the data — contributes nothing.
  `max`/`min`: the gradient goes to the first *selected* extremum, with
  `max`'s tie-break and NaN semantics applied to the selected elements only;
  an empty lane receives none. All four are pinned against finite differences
  by `fucina.gradcheck` ([§5.7](05-automatic-differentiation.md#57-gradient-checking-srcaggradcheckzig)).
- Counting a mask stays `mask.sum(tag, .{})` (i64, [§4.19](04-tensor-operations.md#419-math-on-non-f32-tensors-srcagtensorzig)); there is no separate
  `count` entry.

`max`/`min` cover Fortran's `maxval`/`minval`; `product` and
`all`/`any` deliberately have no masked twin (no in-tree consumer, and
`all`/`any` compose with `logicalAnd`/`logicalOr`).

```zig
test "masked reductions restrict a reduction to the elements a mask selects" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = fucina.Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var x = try fucina.Tensor(.{ .row, .col })
        .variableFromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();
    // row 0 keeps {1, 3}; row 1 keeps {5}
    var mask = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ true, false, true, false, true, false });
    defer mask.deinit();

    var total = try x.sum(&ctx, .col, .{ .mask = &mask });
    defer total.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 4, 5 }, try total.dataConst());

    // The mean divides by the SELECTED count (2 and 1), not the axis length.
    var avg = try x.mean(&ctx, .col, .{ .mask = &mask });
    defer avg.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 5 }, try avg.dataConst());

    // Gradient reaches only the selected elements.
    var loss = try total.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?;
    defer gx.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 1, 0, 1, 0 }, try gx.dataConst());
}

test "a masked lane that selects nothing takes the identity, not an error" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = fucina.Tensor(.{ .dtype = .bool, .tags = .{ .row, .col } });
    var x = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var mask = try M.fromSlice(&ctx, .{ 2, 2 }, &.{ false, false, true, false });
    defer mask.deinit();

    var total = try x.sum(&ctx, .col, .{ .mask = &mask });
    defer total.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 3 }, try total.dataConst()); // identity, no error

    // A mean has no identity: 0/0 unless the caller names a sentinel.
    var avg = try x.mean(&ctx, .col, .{ .mask = &mask, .empty = 0 });
    defer avg.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0, 3 }, try avg.dataConst());
}
```

Related (models band): `kdaRecurrent`
(`src/models/research/kimi3/delta_attention.zig`) is the fused, stateful
counterpart of this scan family — the delta-rule linear-attention
recurrence the kimi3 model family mixes sequences with; it lives with its
family, described under [§4.13](04-tensor-operations.md#413-attention-srcagtensorzig).

Dtype policy: on the f32 facade everything is f32 in and out. On the typed
constant tensors ([§4.19](04-tensor-operations.md#419-math-on-non-f32-tensors-srcagtensorzig)) reductions widen per `outputDType(.reduction, ·)` —
f16/bf16 reduce into f32, f64 stays f64 — while pointwise and matmul keep
the input dtype; see [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md) for the full dtype/storage matrix.

```zig
test "axis reductions and sumAll" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .row, .col }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var s = try x.sum(&ctx, .col, .{}); // Tensor(.{ .row })
    defer s.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 7 }, try s.dataConst());
    var v = try x.variance(&ctx, .col, .{ .ddof = 0 }); // biased
    defer v.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.25 }, try v.dataConst());
    var total = try x.sumAll(&ctx); // Tensor(.{}) scalar
    defer total.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 10), try total.item(), 1e-6);
}

test "cumsum keeps the axis" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 1, 2, 3 });
    defer x.deinit();
    var y = try x.cumsum(&ctx, .d);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 6 }, try y.dataConst());
}

test "linearRecurrence scans a decayed state along the time tag" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // h_t = a[d]·h_{t-1} + b_t per lane; decay broadcasts by tag.
    var b = try fucina.Tensor(.{ .t, .d }).fromSlice(&ctx, .{ 3, 2 }, &.{ 1, 1, 1, 1, 1, 1 });
    defer b.deinit();
    var a = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 0.5, 0 });
    defer a.deinit();
    var h = try b.linearRecurrence(&ctx, .t, &a, .{});
    defer h.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1.5, 1, 1.75, 1 }, try h.dataConst());

    // .initial seeds h_{-1} per lane (the chunked-streaming seam).
    var h0 = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 8, 8 });
    defer h0.deinit();
    var seeded = try b.linearRecurrence(&ctx, .t, &a, .{ .initial = &h0 });
    defer seeded.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 1, 3.5, 1, 2.75, 1 }, try seeded.dataConst());
}
```

## 4.8 `dot`: tag-directed contraction (`src/ag/tensor.zig`, `src/tag_ops.zig`)

```zig
pub fn dot(self: *const Self, ctx: *ExecContext, other: anytype, comptime contract_tag: Tag)
    !Tensor(dotResultTags(tags, TensorObject(@TypeOf(other)).axis_tags, contract_tag))
// TensorObject unwraps pointer operands (`a.dot(&ctx, &b, .k)`)
```

`dot` is the workhorse contraction: it sums over the **named** tag
`contract_tag`, which must appear in both operands with equal dims. Tag
roles are decided at comptime:

- **contract** — `contract_tag`; removed from the result.
- **batch** — every other tag shared by both operands; dims must match
  exactly (batch tags do not broadcast).
- **free** — tags private to one operand.

The result tag order is `batch ++ left-free ++ right-free` (each group in
its operand's order). Because tags name axes, no `transpose` calls are ever
needed around `dot` — layout is handled by the lowering.

**Lowering**: `dot` is the single-contract-tag special case of `einsum` —
`taggedDot` delegates to `taggedEinsum` with the canonical dot result order
as the equation, so kernel selection is the einsum lowering's ([§7.9](07-named-axes-the-tag-algebra.md#79-taggeddot-tag-directed-contraction-and-its-lowering-srctag_opszig)): each
operand is aligned to the kernel layout as a zero-copy view, and at runtime
each side picks plain or transposed orientation by contiguity, so classic
layouts (`[m,k]·[k,n]`, NT weights `[out,in]`, batched `[b..,m,k]·[b..,k,n]`
and their trans permutations) dispatch straight to
`matmul2D`/`matmulTransA`/`matmulTransB`/`bmm`/`bmmTransA`/`bmmTransB` with
no data movement, vector operands ride along as size-1 GEMM axes, and a
full contraction runs `ctx.dot`. At most one of transA/transB is available
per call, so a layout where BOTH operands want their transposed orientation
(and no output-order swap covers it) materializes the smaller operand;
layouts no orientation can express materialize at most once per operand.

**Mixed-precision and quantized weights.** `other`'s dtype is
comptime-dispatched:

- f32 RHS: full two-operand backward (`DotBackward`).
- f16 / bf16 RHS (`ConstRhsDotBackward`): a CONSTANT RHS is a frozen weight
  — gradient flows to `self` only; a grad-requiring 16-bit RHS **variable**
  ([§5.1](05-automatic-differentiation.md#51-the-gradient-model-srcagtensorzig-srcagcorezig)) also receives its own gradient, as f32 (gradients are always f32;
  dW is the plain f32 einsum of the upstream gradient with the saved f32
  LHS). With no batch tags, one RHS free axis, and RHS storage
  `[free, contract]` (weight tags `{.out, .in}`-style) the forward hits the
  dedicated trans-B mixed kernels that widen in-register; otherwise the RHS
  is cast to f32 once and the f32 path runs.
- block-quantized RHS (q8_0, q4_k, ... — [§10](10-quantization.md)): the quantized-RHS GEMM;
  gradient to `self` only. Requires RHS storage `[free, contract]`, one RHS
  free axis, no batch tags (compile errors otherwise). When a GPU backend is
  active the GEMM may be offloaded ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)) with or without gradients — the LHS
  gradient flows through the dequantized weight either way; checkpoint
  blocks disable the offload for both of their runs ([§5.5](05-automatic-differentiation.md#55-activation-checkpointing-srcagcheckpointzig)).

```zig
test "dot with a shared batch tag lowers to bmm" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var a = try fucina.Tensor(.{ .b, .m, .k }).fromSlice(&ctx, .{ 2, 1, 2 }, &.{ 1, 2, 5, 6 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .b, .k, .n }).fromSlice(&ctx, .{ 2, 2, 1 }, &.{ 3, 4, 7, 8 });
    defer b.deinit();
    var y = try a.dot(&ctx, &b, .k); // .b is shared (batch), result .{ .b, .m, .n }
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 83 }, try y.dataConst());
}

test "dot with an f16 constant RHS stays mixed-precision" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .t, .in }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 1 });
    defer x.deinit();
    var w = try fucina.Tensor(.{ .dtype = .f16, .tags = .{ .out, .in } })
        .fromSlice(&ctx, .{ 1, 2 }, &.{ 2, 3 });
    defer w.deinit();
    var y = try x.dot(&ctx, &w, .in); // f32 result, [free, contract] fast path
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{5}, try y.dataConst());
}
```

### `addDot`: fused residual/projection accumulate (addmm)

```zig
pub fn addDot(self: *const Self, ctx: *ExecContext, a: anytype, b: anytype, comptime contract_tag: Tag) !Self
```

`self + a·b` in one op — torch's `addmm`. The product accumulates directly
into a copy of `self` (the BLAS beta=1 route, or the vector accumulate
kernels elsewhere), so the residual pattern costs one GEMM with no
intermediate product tensor and no separate add pass, and backward is one
node (the base gradient is the output gradient itself). On the vector
kernels the result is bit-identical to `a.dot(b)` then `add`; the BLAS
route may differ in the last ulp, as any two BLAS entry points may.

The form is compile-checked: f32 operands, `a` tagged `[rows, contract]`,
`b` tagged `[contract, cols]`, `self` tagged exactly `[rows, cols]`.
Differentiable in all three operands; other RHS dtypes compose `dot` +
`add` instead.

```zig
test "addDot fuses the residual add into the GEMM" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var residual = try fucina.Tensor(.{ .t, .out }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 1, 2, 2 });
    defer residual.deinit();
    var a = try fucina.Tensor(.{ .t, .in }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var w = try fucina.Tensor(.{ .in, .out }).fromSlice(&ctx, .{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer w.deinit();

    var y = try residual.addDot(&ctx, a, w, .in); // residual + a·w, one GEMM
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 20, 23, 45, 52 }, try y.dataConst());
}
```

### `einsum` and `einsumMany`: multi-index contraction

```zig
pub fn einsum(self: *const Self, ctx: *ExecContext, other: anytype, comptime out_tags: anytype)
    !Tensor(normalizeTags(out_tags))
pub fn einsumMany(ctx: *ExecContext, comptime out_tags: anytype, operands: anytype)
    !Tensor(normalizeTags(out_tags))   // free function at the fucina root
```

`einsum` generalizes `dot` from one contraction tag to a whole Einstein
equation. Because both operands already carry named axes, the output tag
tuple **is** the equation:

```
result[out_tags] = Σ over every tag not in out_tags of self ⊙ other
```

Tag roles are decided at comptime purely from membership:

- **batch** — shared tags kept in `out_tags`; dims must match exactly.
- **contract** — shared tags dropped from `out_tags`; dims must match
  (`ShapeMismatch` otherwise). Any number of contraction tags.
- **free** — operand-private tags kept in `out_tags`.
- **summed** — operand-private tags dropped from `out_tags`; summed away
  before the contraction (their gradient is a broadcast).

The result axis order is exactly `out_tags` (unlike `dot`, whose result
order is fixed); every output tag must exist in an operand (compile error
`einsum output tag not found in any operand`). `self` is f32; an f16/bf16 `other`
is widened to f32 once per call (forward and backward) — a constant RHS
routes gradient to `self` only, and a grad-requiring 16-bit RHS variable
also receives its own f32 gradient, exactly dot's widened fallback
contract — and
a quantized `other` is a compile error directing to `dot`, whose packed
kernels require the `[free, contract]` weight layout. Duplicate tags within
one operand remain impossible, so there are no trace/diagonal semantics. `dot(other, .k)` and
`einsum(other, dotResultTags(...))` compute the same thing; `einsum` is the
one to reach for the moment an equation has several contraction axes,
several free axes on one side, or a specific output order.

**Lowering** (`taggedEinsum`, [§7.9](07-named-axes-the-tag-algebra.md#79-taggeddot-tag-directed-contraction-and-its-lowering-srctag_opszig)): summed-private axes are pre-reduced,
then both operands are aligned (zero-copy permute views) to an
`out_tags`-derived group-nested order and each side independently picks the
plain or transposed kernel layout at runtime — the orientation whose aligned
view is already contiguous wins, because a trans GEMM is free while
materializing costs a copy pass. Classic layouts therefore dispatch to
`matmul2D`/`bmm` (and trans variants) with zero copies; at most one of
transA/transB is available per call (both-want-trans materializes the
smaller operand), and layouts no orientation can express materialize at
most once per operand. When
`out_tags` nests as `[batch][right free][left free]`, the operands swap
kernel roles, so "double-transposed" layouts (e.g. `x[k,m] · y[n,k] ->
[n,m]`) run as one plain GEMM with zero copies. An `out_tags` order that
interleaves the three groups costs one extra output materialization —
prefer group-nested orders.

**Backward.** Contractions are closed under differentiation: each operand's
gradient is another einsum (the output gradient contracted with the other
operand), broadcast over any forward-summed axes — so both VJP branches
stay on GEMM kernels for every tag structure (`EinsumBackward`, which
`DotBackward` delegates to, and its const-RHS variant
`ConstRhsEinsumBackward`, which `ConstRhsDotBackward` delegates to). This
retired the old
broadcast-multiply backward fallback for dots with more than one free tag
on the opposite side (`zig build bench-einsum` measured that case at two
orders of magnitude).

`einsumMany` folds two or more operands left-to-right through binary
`einsum`, keeping at each step exactly the tags still needed by the
remaining operands or the output. Contraction order is the operand order —
order the tuple so early intermediates stay small. As with other composed
facade ops, the intermediates are released on return and retained by the
records, so gradients flow with or without an exec scope.

```zig
test "einsum: one equation for a grouped-attention-style contraction" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // q[g, i, d] x k[j, d] -> scores[g, i, j]: two free axes on the left and
    // an NT-layout right operand, contracted over .d in one equation.
    var q = try fucina.Tensor(.{ .g, .i, .d }).fromSlice(&ctx, .{ 2, 1, 2 }, &.{ 1, 2, 3, 4 });
    defer q.deinit();
    var k = try fucina.Tensor(.{ .j, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 0, 0, 1 });
    defer k.deinit();
    var scores = try q.einsum(&ctx, &k, .{ .g, .i, .j });
    defer scores.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, try scores.dataConst());
}

test "einsum: dropped tags are summed — shared become contractions, private are pre-summed" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // y[n] = sum over s and k of a[s,k] * b[k,n]: .k is shared (contraction),
    // .s is private to `a` and simply summed away.
    var a = try fucina.Tensor(.{ .s, .k }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 2, 3, 4 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .k, .n }).fromSlice(&ctx, .{ 2, 2 }, &.{ 5, 6, 7, 8 });
    defer b.deinit();
    var y = try a.einsum(&ctx, &b, .{.n});
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 62, 72 }, try y.dataConst());
}

test "einsumMany: a LoRA delta as one three-operand equation" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .s, .i }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 1 });
    defer x.deinit();
    var a = try fucina.Tensor(.{ .r, .i }).fromSlice(&ctx, .{ 1, 2 }, &.{ 2, 3 });
    defer a.deinit();
    var b = try fucina.Tensor(.{ .o, .r }).fromSlice(&ctx, .{ 2, 1 }, &.{ 1, -1 });
    defer b.deinit();

    // x[s,i] · A[r,i] · B[o,r] -> [s,o], contraction order = operand order.
    var y = try fucina.einsumMany(&ctx, .{ .s, .o }, .{ &x, &a, &b });
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, -5 }, try y.dataConst());
}
```

## 4.9 Explicit matmul, ternary STE, and packed-RHS GEMMs (`src/ag/tensor.zig`)

```zig
pub fn matmul(self, ctx, other: anytype, comptime kind: exec.MatmulKind, comptime out_tags: anytype)
    !Tensor(out_tags)
```

`matmul` bypasses the tag algebra: the caller names the result axes and
picks `exec.MatmulKind` (`.plain`, `.trans_a`, `.trans_b`). Routing is
comptime on rank: both operands rank-2 → the 2-D GEMM entries (`.plain`:
`[m,k]·[k,n]`; `.trans_b`: `[m,k]·[n,k]ᵀ`); anything else → the batched bmm
entries with stride-0 broadcast leading batch axes (mixed-rank operands
broadcast rather than error). `.trans_a` exists only on the batched path —
rank-2 `.trans_a` is a compile error directing to `dot`, whose tag algebra
reaches the 2-D trans-A kernel. f32 only, full two-operand gradients. Unlike
`dot` there is no materialize fallback: the operands' storage order **is**
the kernel layout.

```zig
test "explicit matmul with a transposed RHS" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .m, .k }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 2 });
    defer x.deinit();
    var w = try fucina.Tensor(.{ .n, .k }).fromSlice(&ctx, .{ 2, 2 }, &.{ 3, 4, 5, 6 });
    defer w.deinit();
    var y = try x.matmul(&ctx, &w, .trans_b, .{ .m, .n }); // [m,k] · [n,k]ᵀ
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 11, 17 }, try y.dataConst());
}
```

**`dotTernarySte(ctx, weight, contract_tag)`** — trainable ternary linear
(BitNet b1.58 straight-through estimator). Every forward encodes the f32
latent `weight` (tags `{.out, .in}`, per-tensor absmean scale, round-clip to
{−1, 0, +1}) to TQ2_0 and contracts with the mul-free kernel. Backward: `dx`
flows through the **quantized** weight; `dW` is the straight-through
estimate (plain matmul VJP against the latent weight). The contract dim must
be a multiple of 256 (`error.TernaryContractDimNotBlockAligned` otherwise);
no shared batch tags, one weight free axis, weight storage
`[free, contract]`, and lhs storage `[..., contract]` (contract tag last) —
all four are compile errors. See `TERNARY.md` and [§10](10-quantization.md)/[§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md).

```zig
test "dotTernarySte encodes the latent weight per call" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const k = 256; // contract dim must be a multiple of the TQ2_0 block size
    const x_values = [_]f32{1} ** k;
    var x = try fucina.Tensor(.{ .t, .in }).fromSlice(&ctx, .{ 1, k }, &x_values);
    defer x.deinit();
    var w = try fucina.Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 1, k }, &x_values);
    defer w.deinit();
    var y = try x.dotTernarySte(&ctx, &w, .in); // absmean scale 1, all-ones encode
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{k}, try y.dataConst());
}
```

**Packed-RHS entries** (layout and quantization detail in [§10](10-quantization.md)):

- `dotPacked(ctx, rhs, contract_tag, out_tag)` — 2-D `[free, contract]` lhs
  against a pre-packed dense or quantized RHS container (comptime-dispatched
  from the pointer type). Dense f32/f16/bf16 packs fail with
  `error.GradientPackedMatmulUnsupported` when the lhs requires grad;
  quantized packs fail with `error.GradientQuantizedMatmulUnsupported`.
- `rmsNormMulDotPacked(ctx, norm_weight, eps, rhs, contract_tag, out_tag)` —
  fused `rmsNormMul(self, norm_weight) · rhsᵀ` without materializing the
  normalized tensor (`self` is the pre-norm `[free, contract]` input,
  `norm_weight` the `[contract]` scale row); matches `rmsNormMul` +
  `dotPacked` to ≤ 1 ulp (q8_0x4 / q4_kx8 / q5_kx8 / q6_kx4; q4_kx2mmla is
  a deliberate compile error — MMLA targets use the unfused path).
- `splitSwiGluDotPacked(ctx, rhs, split_tag, out_tag)` — fused split-SwiGLU
  + packed down-projection GEMM without materializing the gated tensor
  (q8_0x4 / q4_kx8 / q5_kx8 / q6_kx4; q4_kx2mmla is a deliberate compile
  error — MMLA targets use the unfused `splitGated` + `dotPacked`).
- `gegluQuantDotPacked(ctx, up, rhs, in_tag, out_tag)` — fused
  `(up · geluQuant(self)) @ rhs`; q8_0x4 only.
- On block-quantized tensors: `packRhs(ctx)` packs a rank-2 weight into the
  ISA-best layout for its dtype — q8_0→x4, q6_k→x4, q5_k→x8, q4_k→x2mmla on
  aarch64+i8mm targets else x8 (the return type is
  `fucina.PackedRhs(dtype)`); `packRhsAs(ctx, Rhs)` forces a specific
  container type (`fucina.quant.QuantizedMatmulRhsQ4_Kx8`, ...) instead.
- On f32/f16/bf16 tensors: `packRhs(ctx)` snapshots a rank-2 `[out, contract]`
  weight into the shared f32 output-row-panel layout. f16/bf16 values widen
  once while packing; the caller owns and `deinit()`s the returned
  `fucina.PackedRhs(dtype)`. The source tensor may be released immediately.

```zig
test "load-time dense packed RHS" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const w_values = [_]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 1,
    };
    var packed_rhs = blk: {
        var w = try fucina.Tensor(.{ .out, .in }).fromSlice(&ctx, .{ 3, 4 }, &w_values);
        defer w.deinit();
        break :blk try w.packRhs(&ctx); // independent snapshot of w
    };
    defer packed_rhs.deinit();

    const x_values = [_]f32{ 2, 3, 4, 5, 7, 11, 13, 17 };
    var x = try fucina.Tensor(.{ .seq, .in }).fromSlice(&ctx, .{ 2, 4 }, &x_values);
    defer x.deinit();
    var y = try x.dotPacked(&ctx, &packed_rhs, .in, .out);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, 9, 7, 11, 30 }, try y.dataConst());
}
```

## 4.10 Softmax family (`src/ag/tensor.zig`, `src/exec/softmax.zig`)

```zig
pub fn softmax(self: *const Self, ctx: *ExecContext, comptime tag: Tag, options: anytype) !Self
```

Softmax over `tag`, shape-preserving. `options` is a comptime-validated
struct literal (unknown fields are compile errors); an empty `.{}` routes to
the lean plain kernel. The fused extensions mirror ggml's `soft_max_ext`
(the exec-level option struct is `exec.SoftmaxExtOptions`); the effective
pre-softmax logit is `x·scale + slope·mask`:

- `.scale = s` — logit multiplier (attention `1/sqrt(d)` without a separate
  pass).
- `.mask = &m` — **additive** tag-broadcast tensor (not −inf masking): the
  mask is aligned to `self`'s tags and expanded by zero-stride broadcast, so
  a `[q, k]` mask serves every head of a `[head, q, k]` score tensor. The
  mask must not require grad (`error.UnsupportedGradient`).
- `.max_bias = b` with `.head_tag` — ALiBi: a per-head slope multiplies the
  mask, following the ggml slope schedule (powers of `2^(−b/h)` with `h` the
  head count rounded down to a power of two; `src/exec/softmax.zig`
  `alibiSlope`). Requires `.mask` and `.head_tag` (`InvalidArgument`
  otherwise).
- `.sinks = slice` — per-head attention sinks: one extra logit per head that
  joins the running max and the denominator only, so row probabilities sum
  to less than 1 (the sink absorbs the remaining mass). Needs `.head_tag`
  (one sink per head; a single-element slice is accepted without one).
- `.causal = .{ .query_tag, .source_offset }` — fused causal masking: query
  row `q` normalizes over sources `[0, source_offset + q]`; positions beyond
  are exactly 0. Validates `source_offset + query_dim <= source_dim`.

Backward: `SoftmaxBackward`/`SoftmaxExtBackward` ([§5](05-automatic-differentiation.md)) — the unified backward
re-derives from the output and `scale` (mask/sinks/ALiBi contribute no
gradient).

Two log-domain companions (max-shifted for stability, torch semantics),
FUSED single-node kernels sharing softmax's row machinery — SIMD max scan
+ vexpf sum per row, task-parallel over rows, streaming inner-lane kernels
on non-last axes (lane-split across tasks, identical semantics — [§9.4](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#94-native-backend-portable-vector-kernels-srcbackendvector));
no materialized intermediates and no exec-scope requirement:

- `logsumexp(ctx, tag)` — `log(Σ exp(x))` with the tag removed
  (torch.logsumexp). Rows whose max is ±inf are shifted by 0 instead, so
  an all(-inf) row yields -inf and a +inf entry yields +inf rather than
  NaN (the torch convention). Backward is the saved-output identity
  `exp(x − lse)·g` (the row softmax).
- `logSoftmax(ctx, tag)` — `x − logsumexp(x)` broadcast, shape-preserving
  (torch.log_softmax), same non-finite-max handling. Prefer
  `crossEntropy` when the next step is an NLL loss (fused with the loss,
  saved-stats backward). Backward is the saved-output identity
  `g − exp(y)·Σg`.

```zig
test "softmax and the fused causal extension" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .seq, .src }).zeros(&ctx, .{ 2, 2 });
    defer x.deinit();
    var p = try x.softmax(&ctx, .src, .{});
    defer p.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.5, 0.5, 0.5 }, try p.dataConst());

    var c = try x.softmax(&ctx, .src, .{ .causal = .{ .query_tag = .seq } });
    defer c.deinit();
    // query 0 attends source 0 only; masked-out tail is exactly 0
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0.5, 0.5 }, try c.dataConst());
}
```

## 4.11 Normalization family (`src/ag/tensor.zig`)

All norms normalize over one named tag and preserve shape; all are
differentiable in every tensor operand (statistics are recomputed in the
backward — nothing extra is saved from the forward).

- `rmsNorm(ctx, tag, eps)` — `x / sqrt(mean(x²) + eps)`.
- `rmsNormMul(ctx, tag, weight, eps)` — fused `rmsNorm(x)·weight`;
  `weight: *const Tensor(.{tag})`.
- `rmsNormMulAdd(ctx, tag, weight, residual, eps)` — fused
  `rmsNorm(x)·weight + residual` (`residual` same tags as `self`).
- `rmsNormMulRopeHalfPrepared(ctx, position_tag, feature_tag, weight, eps, table)`
  — fused rmsNorm·weight followed by half-mode RoPE from a prepared
  `*const exec.RopeTable` (the QK-norm + RoPE step of the model families'
  attention blocks, [§14](14-model-families-and-example-applications.md)). The inference-only `rmsNormMulDotPacked`
  (rmsNormMul fused into a packed quantized GEMM) is documented with the
  packed-RHS entries in [§4.9](04-tensor-operations.md#49-explicit-matmul-ternary-ste-and-packed-rhs-gemms-srcagtensorzig).
- `layerNorm(ctx, tag, eps, options)` — PyTorch semantics:
  `(x − μ)/sqrt(σ² + eps)` with biased variance. `options` is `.{}` for the
  plain form or `.{ .weight = &w, .bias = &b }` for the fused affine — the
  fused kernel requires **both** together (compile error otherwise).
  Weight/bias are rank-1 `[tag_dim]` tensors, either tagged `.{tag}`
  (comptime-checked) or numeric-tag `Tensor(1)` values (`._0`; runtime
  length check).
- `groupNorm(ctx, channel_tag, groups, eps, options)` — ggml GroupNorm
  over rank-2 `[time, channel]` storage: per channel group, f64-accumulated
  mean and biased variance over all `time × (C/groups)` elements, eps inside
  the sqrt, then the optional per-channel affine applied after
  normalization (`options: GroupNormOptions(channel_tag)`: `.{}` plain, or
  `.{ .weight = &w, .bias = &b }` with rank-1 `.{channel_tag}` tensors,
  independently optional).
- `standardizeAxis(ctx, tag, options)` — `(x − mean)/denom` over `tag`.
  `options` accepts every `exec.StandardizeOptions` field plus an optional
  `.valid_len` (unknown fields are compile errors): standardize only the
  first `valid_len` elements; the suffix is returned as zeros and receives
  zero gradient. `StandardizeOptions`: `ddof: u1 = 0`, `eps: f32 = 0`,
  `eps_mode: StandardizeEpsMode = .outside_sqrt` (`sqrt(var) + eps`,
  Parakeet's frontend convention) or `.inside_sqrt` (`sqrt(var + eps)`,
  LayerNorm placement), `accumulation: StandardizeAccumulation = .f32` or
  `.f64`.
- `l2Normalize(ctx, tag, eps)` — `x·rsqrt(Σx² + eps)`. The eps is added to
  the **squared** norm (the rmsNorm convention) — deliberately not torch
  `F.normalize`'s `x/max(‖x‖₂, eps)`. Composed op: requires an active exec
  scope when gradients are tracked ([§4.1](04-tensor-operations.md#41-the-common-operation-contract-srcagtensorzig)).
- `snake(ctx, channel_tag, alpha, inv_b)` — per-channel Snake activation
  (DAC codec): `y = x + inv_b[c]·sin(alpha[c]·x)²` over `[time, channel]`
  storage. `alpha` and `inv_b` are independent operands at this level; no
  gradient flows through the loader's `inv_b = 1/(alpha + 1e-9)` relation.

```zig
test "layerNorm and rmsNorm" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .t, .d }).fromSlice(&ctx, .{ 1, 2 }, &.{ 1, 3 });
    defer x.deinit();
    var ln = try x.layerNorm(&ctx, .d, 0, .{}); // (x - mean) / sqrt(biased var + eps)
    defer ln.deinit();
    try std.testing.expectEqualSlices(f32, &.{ -1, 1 }, try ln.dataConst());

    var rn = try x.rmsNorm(&ctx, .d, 0); // x / sqrt(mean(x²) + eps)
    defer rn.deinit();
    const rms = @sqrt((1.0 + 9.0) / 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 / rms), (try rn.dataConst())[1], 1e-6);
}

test "standardizeAxis zero-mean unit-variance" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .t, .d }).fromSlice(&ctx, .{ 1, 4 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var y = try x.standardizeAxis(&ctx, .d, .{}); // ddof 0, eps 0
    defer y.deinit();
    const out = try y.dataConst();
    try std.testing.expectApproxEqAbs(@as(f32, 0), out[0] + out[1] + out[2] + out[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5 / @sqrt(1.25)), out[3], 1e-6);
}
```

## 4.12 Rotary position embedding (`src/ag/tensor.zig`, `src/exec/rope.zig`)

```zig
pub fn rope(self, ctx, comptime position_tag: Tag, comptime feature_tag: Tag,
            source: anytype, comptime mode: RopeMode) !Self
```

Rotates feature pairs by position-dependent angles over
(`position_tag`, `feature_tag`). `mode` is comptime `exec.RopeMode`:
`.half` pairs feature `i` with `i + d/2` (NEOX/Llama layout);
`.interleaved` pairs adjacent features; `.interleaved_tail` pairs adjacent
features within the TRAILING `table.feature_dim` features — a partial
rotary aligned to the end of the feature axis with the leading features
passed through (DeepSeek V4's tail-64 rotary; identical to `.interleaved`
when the table spans the whole axis). `source` selects the factor source
at comptime (a closed set; anything else is a compile error):

- `*const exec.RopeTable` — prepared factors, the production path. Build
  with `ctx.prepareRopeTable(spec)`, where `exec.RopeTableSpec` names the
  positions, the rotary span, the angle schedule and the sign:

  ```zig
  pub const RopeTableSpec = struct {
      positions: RopePositions,   // .explicit = []const i32 | .range = AxisRange
      feature_dim: usize,         // the table's rotary span
      freqs: RopeFreqs,           // .theta = .{ .base, .factors = null } | .inv_freq_f64 = []const f64
      inverse: bool = false,      // negate sin: the un-rotation table
  };
  ```

  `.theta` computes `pos / base^(2i/d)` in f32; `factors` (length
  `feature_dim/2`) divides each pair's frequency: ggml's `rope_ext`
  `freq_factors` (Llama-3 long-context, Gemma global layers); `null` is
  plain RoPE. `.inv_freq_f64` takes per-pair inverse frequencies the core
  cannot rebuild and accumulates each angle in f64 before the f32 cast;
  the model-band `models.ops.yarnBlendInvFreqsF64(ctx, dim, base, factor,
  orig_ctx)` (`src/models/ops.zig`) builds the DeepSeek-family YaRN blend
  for it (beta 32/1 correction ramp; `factor <= 1` returns the plain pow
  schedule). The table's `feature_dim`
  is the **authoritative rotary span**: equal to `dim(feature_tag)` rotates
  fully; smaller rotates the leading `feature_dim` features (`.half`,
  `.interleaved`) or the trailing ones (`.interleaved_tail`) and passes the
  rest through unchanged (partial RoPE). `RopeTable` owns its buffers
  through a shared owner count: `table.deinit()` releases one handle (the
  buffers go with the last), and `table.retain()` returns another owning
  handle of the same buffers — how the RoPE VJP records keep the forward
  table alive without copying it.
- `exec.RopeTheta` / `.{ .positions = p, .theta_base = t }` — on-the-fly
  factors (`positions: []const i32`), full rotation only. Pair `i` at
  position `p` rotates by `p / theta_base^(2i/d)`.

Differentiable in `self`; the backward applies the inverse rotation ([§5](05-automatic-differentiation.md)).
Negative positions rotate backwards, so re-roping cached values to a new
offset is a valid pattern.

**Position runs: `AxisRange`.** A tensor axis here is 0-origin: `Shape`
records how long an axis is, never where it starts. A positional axis has an
origin anyway — the absolute token position of its first row — and with
nowhere to put it, callers materialized it: allocate `n` integers, fill them
with `origin + i`, pass them down, free them, all to express `origin`.
`fucina.AxisRange` is that origin as a value (Fortran's array lower bound,
which NumPy and torch have no equivalent of):

```zig
pub const AxisRange = struct {
    origin: i64 = 0,   // absolute index of the axis's first element
    len: usize,        // the axis spans [origin, origin + len)
    pub fn at(self, i: usize) i64
    pub fn end(self) i64
    pub fn narrowed(self, start: usize, count: usize) AxisRange  // keeps the absolute origin
    pub fn shifted(self, delta: i64) AxisRange
    pub fn rebased(self, new_origin: i64) AxisRange
    pub fn contains(self, absolute: i64) bool
    pub fn writeInto(self, out: []i32) !void   // the materialization it avoids
};
```

The rope-table spec takes one as its positions source:

```zig
ctx.prepareRopeTable(.{ .positions = .{ .range = .{ .len = n } }, .feature_dim = d, .freqs = .{ .theta = .{ .base = 10000 } } })
```

`.{ .len = n }` is a prefill over `0..n`; `.{ .origin = pos0, .len = n }` is a
decode step. Both position sources feed one arithmetic body, so a run
expressed as a range and the same run expressed as an array produce
**bitwise identical** tables (pinned in `src/exec/rope_tests.zig` across several
origins, both `inverse` arms, and the `factors` arm). The
`.explicit = []const i32` source stays for the genuinely ragged case, a
multi-stream batch whose positions are several runs, not one.

**Band masks already carry an origin.** An offset causal mask needs no new
API: a row origin shifts a `bandMask`/`bandPart` band by folding into the
bounds. Queries at absolute `p0..p0+n` over keys `0..k` keep
`absolute_q >= absolute_k`, which is `bandMask(shape, null, p0)` — the
`upper` bound absorbs the origin, since `(i + p0) - j >= 0` is `j - i <= p0`.
Generally, an origin difference `o = row_origin - col_origin` turns bounds
`(lower, upper)` into `(lower - o, upper + o)`.

```zig
test "rope rotates feature pairs by position" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .seq, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 1, 0, 1, 0 });
    defer x.deinit();
    var y = try x.rope(&ctx, .seq, .d, .{ .positions = &.{ 0, 1 }, .theta_base = 10000 }, .half);
    defer y.deinit();
    const out = try y.dataConst();
    try std.testing.expectApproxEqAbs(@as(f32, 1), out[0], 1e-6); // position 0: identity
    try std.testing.expectApproxEqAbs(@cos(@as(f32, 1)), out[2], 1e-6); // position 1: angle 1 rad
    try std.testing.expectApproxEqAbs(@sin(@as(f32, 1)), out[3], 1e-6);
}
```

## 4.13 Attention (`src/ag/tensor.zig`)

```zig
pub fn groupedAttention(self, ctx, k: anytype, v: anytype, kv_head_for_head: []const usize,
                        comptime out_tag: Tag, scale_value: f32, opts: anytype)
    !Tensor(.{ .seq, out_tag })
```

Grouped-query (GQA) flash-style attention. `self` is the query and **must**
be tagged `.{ .seq, .head, .d }` (compile error otherwise); the result is
`[seq, head·d]` tagged `.{ .seq, out_tag }`. `kv_head_for_head[h]` maps each
query head to its KV head. `scale_value` multiplies the scores. The KV
representation is comptime-dispatched from `@TypeOf(k)` (k and v must
match):

| `k`/`v` type | Use case | Gradients |
|---|---|---|
| `*Tensor(.{ .seq, .kv_head, .d })` (f32) | training / f32 caches | full q/k/v backward (windowed re-masks to the window) |
| same tags, f16 | decode KV cache | q-grad only (K/V widened to f32 once for the backward) |
| `[]const BlockQ8_0` | q8_0 raw-block cache, layout `[kv_seq, kv_heads, d/32]` | q-grad only, causal only |
| `[]const []const f16` | ragged multi-stream decode | inference-only |
| `[]const []const BlockQ8_0` | ragged multi-stream decode (q8_0) | inference-only |

`opts` is a comptime-validated struct literal with a per-representation
field whitelist (a misspelled option is a compile error, never
silently-full-causal attention):

- `.mask = .causal` (default) | `.bidirectional` — f32/f16 KV only.
  Bidirectional has no windowed kernel by design (realize SWA reach by
  narrowing the K/V views).
- `.window = w` — runtime sliding window, 0 = full causal (query `p`
  attends `[max(0, p−w+1), p]`); causal only.
- `.bias = &b` — rank-2 `[q_seq, kv_seq]` additive f32 bias on the scaled
  scores pre-softmax (ggml `soft_max_ext` semantics); bidirectional + f32 KV
  only; inference-only — any grad-requiring operand returns
  `error.UnsupportedGradient`.
- `.kv_seq = n, .kv_heads = h` — required for the q8_0-block representation
  (raw blocks carry no shape).
- `.lens = lens, .kv_heads = h` — required for the multi-stream
  representations; q's `.seq` tag is reinterpreted as the **stream** axis
  (one query row per stream, row `s` attending `lens[s]` cached positions);
  per-stream results are bit-identical to N single-stream calls.

Related: `relposShift(ctx, t_k, out_tags)` — Transformer-XL relative-shift
("skew") of a rank-3 `[H, Tq, P]` score tensor to `[H, Tq, Tk]` with
`out[h,q,j] = self[h, q, j+(Tq−1)−q]` (`P >= Tk+Tq−1`); differentiable
(scatter VJP).

In the models band, `kdaRecurrent`
(`src/models/research/kimi3/delta_attention.zig`, next to its one
consumer) is the fused delta-rule linear-attention sequence recurrence
(Kimi Delta Attention, the kimi3 family's sequence mixer). It takes the
`*ExecContext` first, like an exec entry. Inputs are the post-convolution projections — `q`/`k`
`[seq, heads, k_dim]`, `v` `[seq, heads, v_dim]`, `g_raw`
`[seq, heads, k_dim]` (pre-gate low-rank decay), `beta_raw` `[seq, heads]`
(pre-sigmoid) — plus `a_log` (length `heads` for per-head or `k_dim` for
per-channel decay), `dt_bias` (`heads·k_dim`), an optional
`[heads, k_dim, v_dim]` initial state, and a scale (`<= 0` selects the
default `k_dim^(−1/2)`). Per token, q/k are l2-normalized, each head's
`[k_dim, v_dim]` state decays along K by
`exp(−exp(A_log)·softplus(g + dt_bias))`, the `sigmoid(beta)`-weighted
delta-rule update lands, and the output row is read out. Returns
`delta_attention.KdaResult` — `o` `[seq, heads, v_dim]` plus the
final state (the decode-resume seed), one `deinit()` releasing both. Work
splits by whole heads (single writer per output row — bitwise identical
for any task count); inference-only, no backward record.

```zig
test "groupedAttention over a single cached position returns v" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var q = try fucina.Tensor(.{ .seq, .head, .d }).fromSlice(&ctx, .{ 1, 1, 2 }, &.{ 1, 0 });
    defer q.deinit();
    var k = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 1, 1, 2 }, &.{ 1, 1 });
    defer k.deinit();
    var v = try fucina.Tensor(.{ .seq, .kv_head, .d }).fromSlice(&ctx, .{ 1, 1, 2 }, &.{ 5, 7 });
    defer v.deinit();

    var y = try q.groupedAttention(&ctx, &k, &v, &.{0}, .out, 1.0, .{});
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 5, 7 }, try y.dataConst());
}
```

## 4.14 Convolution and channel-last vision ops (`src/ag/tensor.zig`)

**1-D family** (input storage `[time, channel]`, enforced at comptime; bias
is deliberately not fused — compose it with broadcast `add`):

- `conv1d(ctx, time_tag, in_tag, tap_tag, out_tag, weight, stride, padding, dilation, groups)`
  — PyTorch `Conv1d` semantics (cross-correlation); `weight` is
  `*const Tensor(.{ tap_tag, in_tag, out_tag })` stored `[tap, in/groups, out]`;
  result `[t_out, out]` with
  `t_out = (T + 2·pad − dilation·(taps−1) − 1)/stride + 1`. Differentiable
  in input and weight.
- `causalConv1d(ctx, time_tag, in_tag, tap_tag, out_tag, weight, dilation, state)`
  — causal orientation (tap `taps−1` is the newest sample); `state`, when
  given, supplies the `dilation·(taps−1)` input rows preceding `x` (oldest
  first, `[row, in]`); absent rows read as zeros; no gradient into `state`.
- `groupedCausalConv1d(ctx, time_tag, in_tag, tap_tag, in_per_group_tag, out_tag, weight, dilation, groups, state)`
  — grouped variant, weight `[tap, in_per_group, out]`.
- `causalDepthwiseConv1d(ctx, time_tag, channel_tag, tap_tag, kernel, dilation, state)`
  — depthwise; `kernel: *const Tensor(.{ channel_tag, tap_tag })` (tap
  `taps−1` = the newest sample). `dilation` spaces the taps
  (`y[t,c] = Σ_k x[t − dilation·(taps−1−k), c]·w[c,k]` — Engram's
  ShortConv is `dilation = max_ngram_size`); `state`, when given,
  supplies the `dilation·(taps−1)` rows preceding `x`, oldest first.
- `convTranspose1d(ctx, time_tag, in_tag, kout_tag, out_tag, weight2, bias, out_channels, taps, stride, padding, output_pad)`
  — GEMM + col2im_1d (ggml decomposition); `weight2` is the load-time
  repacked `[K·OC, IC]` matrix (k fastest within each oc block), `bias` is
  `?*const Tensor(.{out_tag})`. The `output_pad` trailing rows are
  bias-only (ggml/omnivoice convention, not exact PyTorch when pad > 0).
  Differentiable in input, weight2, and bias; the weight gradient is with
  respect to the **packed** layout.

**Channel-last 2-D family** (rank-3 `[H, W, C]` inputs; used by [§14](14-model-families-and-example-applications.md)'s face
detection stack):

- `conv2d(ctx, weight, bias, stride, padding, groups, out_tags)` — `weight`
  rank-4 `[Cout, kH, kW, Cin/groups]`, `bias` is `null` or rank-1 `[Cout]`;
  result `[oH, oW, Cout]` tagged `out_tags`. Differentiable in all tensor
  operands.
- `conv2dRelu(...)` — conv2d with the relu fused into the epilogue on the
  no-grad path (identical values to `conv2d` then `relu`; on the Winograd
  route it folds into the output transform). Falls back to the
  differentiable composition when any operand requires gradients.
- `prepareConv2dWeights(ctx)` — on the rank-4 weight: load-time Winograd
  F2/F4 weight-transform planes, built once so the prepared entries below
  skip the per-call weight transform. Returns `fucina.PreparedConvWeights`
  (caller `deinit`s); a weight that can never take the Winograd route
  returns `.empty`, which is inert on every conv route. No gradient
  support (the `dotPacked` policy — prepared planes live outside the
  graph): fails with `error.GradientPreparedConv2dUnsupported` on a
  grad-requiring weight.
- `conv2dPrepared(ctx, weight, prepared, bias, stride, padding, groups, out_tags)`
  — no-grad conv2d against the prepared planes: bitwise-identical values to
  `conv2d`, minus the per-call weight transform on the Winograd route
  (every other route ignores `prepared`). Fails with
  `error.GradientPreparedConv2dUnsupported` when any operand requires grad.
- `conv2dPreparedRelu(...)` — `conv2dPrepared` with the relu fused into the
  conv epilogue; same no-grad contract.
- `maxPool2d(ctx, kernel, stride, padding)` — `[h, w]`-ordered params; the
  zero-pad border reads as −inf. `avgPool2d(...)` averages valid taps only
  (ONNX `count_include_pad=0`). Both differentiable.
- `upsample2xNearest(ctx)` — `[H, W, C] → [2H, 2W, C]`; VJP = 2×2 stride-2
  sum-pool.
- `unfold(ctx, kernel, stride, padding, out_tags)` — patch extraction
  (torch.nn.Unfold in this repo's channel-last layout): rank-3
  `[H, W, C]` → rank-2 `[oH·oW, kH·kW·C]` tagged `out_tags` (patch axis,
  then patch-element axis). Patch row `oy·oW + ox` holds output position
  (oy, ox)'s window taps ordered `(ky, kx, c)` with the channel fastest —
  exactly the im2col layout the groups == 1 conv2d GEMM route consumes,
  so `unfold` → `dot` over the element axis IS a dense conv2d, and a
  stride = kernel, pad 0 unfold is ViT patchify. Out-of-range (pad) taps
  read 0. Shares the threaded im2col kernel (bit-identical to serial);
  differentiable — the VJP is the exact adjoint `fold`.
- `fold(ctx, output_size, kernel, stride, padding, out_tags)` — the
  adjoint (torch.nn.Fold): scatter-ADD rank-2 `[oH·oW, kH·kW·C]` patch
  rows back into a `[H, W, C]` image (three out tags). Overlapping taps
  ACCUMULATE — `fold(unfold(x))` multiplies each position by its window
  count — and pad taps drop; the channel count is derived from the
  patch-element width, and a patch row count that contradicts the
  geometry is `ShapeMismatch`. Shares the threaded col2im kernel;
  differentiable — the VJP is the exact adjoint `unfold`.
- `prelu(ctx, alpha)` — learnable per-channel slope, `alpha` rank-1 `[C]`
  (channel innermost); differentiable in `x` and `alpha`.
- `channelAffine(ctx, scale_t, shift_t)` — fused per-channel
  `x·scale[c] + shift[c]` (frozen-stats inference BatchNorm); differentiable
  in all three.

```zig
test "conv1d computes a moving weighted sum" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .time, .in }).fromSlice(&ctx, .{ 3, 1 }, &.{ 1, 2, 3 });
    defer x.deinit();
    var w = try fucina.Tensor(.{ .tap, .in, .out }).fromSlice(&ctx, .{ 2, 1, 1 }, &.{ 1, 1 });
    defer w.deinit();
    var y = try x.conv1d(&ctx, .time, .in, .tap, .out, &w, 1, 0, 1, 1);
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 5 }, try y.dataConst());
}

test "conv2d channel-last and maxPool2d" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .h, .w, .c }).fromSlice(&ctx, .{ 2, 2, 1 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    var weight = try fucina.Tensor(.{ .oc, .kh, .kw, .ic }).fromSlice(&ctx, .{ 1, 1, 1, 1 }, &.{2});
    defer weight.deinit();
    var y = try x.conv2d(&ctx, &weight, null, .{ 1, 1 }, .{ 0, 0 }, 1, .{ .oh, .ow, .oc });
    defer y.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 6, 8 }, try y.dataConst());

    var pooled = try x.maxPool2d(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 });
    defer pooled.deinit();
    try std.testing.expectEqualSlices(f32, &.{4}, try pooled.dataConst());
}

test "unfold extracts patches and fold accumulates them back" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .h, .w, .c }).fromSlice(&ctx, .{ 2, 2, 1 }, &.{ 1, 2, 3, 4 });
    defer x.deinit();
    // stride = kernel, pad 0: non-overlapping patchify (one 2x2 patch).
    var col = try x.unfold(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 }, .{ .patch, .elem });
    defer col.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, try col.dataConst());
    // fold scatter-adds the patches back (no overlap here → identity).
    var img = try col.fold(&ctx, .{ 2, 2 }, .{ 2, 2 }, .{ 2, 2 }, .{ 0, 0 }, .{ .h, .w, .c });
    defer img.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, try img.dataConst());
}
```

## 4.15 Losses and similarity (`src/ag/tensor.zig`, `src/exec/loss.zig`)

**Cross-entropy** (fused log-softmax + NLL over a named class tag; labels
are `[]const usize`, one per position, positions ordered with the class axis
removed, remaining axes row-major):

```zig
pub fn crossEntropy(self, ctx, comptime class_tag: Tag, labels: []const usize,
                    comptime options: exec.CrossEntropyOptions)
    !Tensor(if (options.reduction == .none) removeTag(tags, class_tag) else .{})
```

`exec.Reduction` is `{ .mean, .sum, .none }`. `exec.CrossEntropyOptions`
(PyTorch parity): `ignore_index: ?usize = null` (matching positions
contribute zero loss/gradient and leave the `.mean` denominator; when
*every* position is ignored the loss is 0, a deliberate divergence from
PyTorch's NaN), `reduction: Reduction = .mean` (`.none` returns per-position
losses with the class tag removed), `label_smoothing: f32 = 0` (in `[0,1)`,
PyTorch semantics). Labels must be `< class_count` or equal to
`ignore_index` (`IndexOutOfBounds` otherwise). Differentiable in the logits.

**Fused linear + cross-entropy** — `crossEntropy(self·weightᵀ)` as ONE
differentiable op:

```zig
pub fn linearCrossEntropy(self, ctx, weight: anytype, labels: []const usize,
                             comptime options: exec.CrossEntropyOptions)
    !Tensor(if (options.reduction == .none) removeTag(tags, tags[1]) else .{})
```

`self` is rank-2 `[row, shared]` and `weight` rank-2 `[class, shared]`
(both f32, shared tag last on both, the class tag absent from `self` —
all comptime-checked). The logits exist only inside the op: they are
computed once and saved on the backward record with the forward's per-row
softmax statistics, and the VJP folds block-built probability panels
straight into dx and dweight, so the `[rows, classes]` logit **gradient**
is never materialized. Differentiable in **both** operands; same
options/reduction contract as `crossEntropy` (`.none` returns per-row
losses tagged by the row tag).

**Fused linear + sparse-soft-target distillation** — cross-entropy of
`self·weightᵀ` against per-entry `(row, class, prob)` soft targets as ONE
differentiable op (a teacher's top-k in the distillation use, [§13.10](13-the-model-stack-fucina_models.md#1310-cartridges-srcmodelstextcartridgezig)):

```zig
pub fn linearDistill(self, ctx, weight: anytype, rows: []const usize,
                        classes: []const usize, probs: []const f32,
                        options: exec.LinearDistillOptions) !Tensor(.{})
```

`loss = reduce_i probs[i]·(LSE(logits[rows[i]]) − logits[rows[i], classes[i]])`
with the same operand contract as `linearCrossEntropy` (rank-2 f32,
shared tag last, comptime-checked). Two structural properties on top of
the fused CE: only the UNIQUE rows named by `rows` are ever projected
(rows without entries produce no logits at all), and the backward consumes
the saved selected-row logits in place, so neither the `[row, class]`
block nor its gradient ever exists outside the record.
`exec.LinearDistillOptions{ .reduction = .mean|.sum, .loss_scale }`
reduces over ENTRIES (`.mean` divides by the entry count; probs are used
as given — a truncated teacher tail is deliberately NOT renormalized) and
`loss_scale` multiplies the scalar result (the gradient-accumulation
knob). Differentiable in **both** operands; the record is single-use like
`linearCrossEntropy` (a repeat backward errors with
`LinearDistillBackwardConsumed`).

**Elementwise losses** vs a same-tagged `target`, all differentiable in
**both** operands, all sharing the reduction/result-type contract above
(`.mean` divides by the **total** element count):

- `mseLoss(ctx, target, options)` — `exec.MseOptions{ .reduction }`.
- `huberLoss(ctx, target, options)` — `exec.HuberOptions{ .delta = 1.0, .reduction }`;
  quadratic for `|x−t| <= delta`, linear beyond.
- `bceLoss(ctx, target, options)` — `exec.BceOptions{ .reduction, .from_logits = false }`.
  With `from_logits` the input is a raw logit and the stable
  `max(x,0) − x·y + log1p(exp(−|x|))` form is used; otherwise a probability
  clamped to `[bce_eps, 1−bce_eps]` (`exec/loss.zig`'s `bce_eps = 1e-7`;
  gradient defined as 0 outside the open interval — deliberate divergence
  from torch's huge boundary gradients).
- `klDivLoss(ctx, target, options)` — `exec.KlDivOptions{ .reduction, .log_target = false }`.
  `self` holds **log**-probabilities; no `.batchmean` exists — `.mean` is
  torch's total-element mean, `.sum` the mathematical divergence.

**Composed** (require an active exec scope when gradients are tracked,
[§4.1](04-tensor-operations.md#41-the-common-operation-contract-srcagtensorzig)):

- `nllLoss(ctx, class_tag, labels, comptime reduction)` — NLL over
  **log-probabilities** (one-hot → mul → sum → negate). Prefer
  `crossEntropy`/`crossEntropy` when starting from logits.
- `cosineSimilarity(ctx, other, tag, eps)` — torch `F.cosine_similarity`:
  `Σxy / max(‖x‖·‖y‖, eps)` with `tag` reduced away; differentiable in both.

```zig
test "crossEntropy on uniform logits is ln(K)" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var logits = try fucina.Tensor(.{ .batch, .class }).zeros(&ctx, .{ 1, 4 });
    defer logits.deinit();
    var loss = try logits.crossEntropy(&ctx, .class, &.{2}, .{});
    defer loss.deinit();
    try std.testing.expectApproxEqAbs(@log(@as(f32, 4)), try loss.item(), 1e-6);

    var per_pos = try logits.crossEntropy(&ctx, .class, &.{2}, .{ .reduction = .none });
    defer per_pos.deinit(); // Tensor(.{ .batch }): class tag removed
    try std.testing.expectApproxEqAbs(@log(@as(f32, 4)), (try per_pos.dataConst())[0], 1e-6);
}

test "mseLoss mean over all elements" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer x.deinit();
    var t = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{2}, &.{ 3, 2 });
    defer t.deinit();
    var loss = try x.mseLoss(&ctx, &t, .{});
    defer loss.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2), try loss.item(), 1e-6); // ((-2)² + 0) / 2
}
```

## 4.16 Selection: argmax, topK, sort, routerTopK (`src/ag/tensor.zig`, `src/exec/topk.zig`)

Index outputs across the library are constant **i64** tensors (the
repo-wide index convention — torch's index dtype, exact for any axis
length) and carry no gradient. As typed constants they are CALLER-owned
even under an exec scope ([§6.3](06-the-execution-runtime-execcontext-and-the-memory-model.md#63-exec-scopes-implicit-ownership-for-training-srcexeczig-srcexecruntimezig)): pair them with `deinit` — an f32
`values` arm of the same call remains a scope-owned borrow.

- `argmax(ctx, tag)` — indices of the per-row maximum, tag removed, as
  `Tensor(.{ .dtype = .i64, .tags = ... })`. NaN never wins (comparisons
  drop NaN): the winner is the maximum over the non-NaN elements, and an
  all-NaN row falls back to index 0 — documented divergence from
  `torch.argmax`, which propagates NaN. **No-grad by design** (like
  sampling).
- `topK(ctx, tag, k, out_tag)` — returns `TopKResult(replaceTag(tags, tag, out_tag))`,
  a struct of `values` and `indices` with a single `deinit()` releasing
  both. NaN never places (it fails the descending-slot admission test); a
  row with fewer than `k` non-NaN elements leaves its unfilled tail slots
  at value −inf, index 0 — documented divergence from `torch.topk`, which
  treats NaN as greater than every number. `values` **is** differentiable
  (the gradient scatters back through the saved indices); `indices` is a
  constant i64 tensor.
- `sort(ctx, tag, descending)` — full sort (`TopKResult(tags)`): values +
  source index per output position. **Unstable** sort; NaN sorts **last**
  regardless of direction (documented divergence from `torch.sort`, which
  puts NaN first when descending). Values differentiable, indices constant.
- `argsort(ctx, tag, descending)` — the indices arm alone (i64); no-grad.
- `multinomial(ctx, class_tag, out_tag, num_samples, seed, replacement)` —
  categorical sampling from UNNORMALIZED non-negative weight rows
  (torch.multinomial): `num_samples` i64 draws per row along `class_tag`
  (last axis, like `routerTopK`; rank 1 or 2 — the torch surface),
  `class_tag` replaced by `out_tag` in the result. Draw `(row, s)` reads
  the deterministic counter-based stream at `(seed, row·num_samples + s)`
  ([§6.8](06-the-execution-runtime-execcontext-and-the-memory-model.md#68-determinism-and-the-rng-contract-srcrngzig)) — reproducible and batching-independent; pass a fresh seed per
  call. Rows must hold finite weights `>= 0` with a positive sum
  (`InvalidShape` otherwise); zero-weight classes are never drawn.
  Without replacement each draw removes the chosen class's mass, and
  `num_samples` beyond the class count — or beyond a row's nonzero
  classes — is `InvalidShape`. **No-grad by design** (an i64 sampling
  output, like `argmax`).
- `routerTopK(ctx, expert_tag, k, options, selected, weights)` — the MoE
  router primitive: fills caller-provided `selected: []usize` /
  `weights: []f32` (both `rows·k` long) with the per-row top-k experts and
  their softmax probabilities computed over the **full** expert axis.
  `exec.RouterTopKOptions{ .normalize_selected: bool = true }` renormalizes
  the selected mass to sum to 1. Requires rank-2 `[row, expert]` logits with
  the expert tag last (compile errors) and a no-grad input
  (`error.UnsupportedGradient`).

```zig
test "argmax, topK, and routerTopK" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var logits = try fucina.Tensor(.{ .row, .expert }).fromSlice(&ctx, .{ 1, 4 }, &.{ 1, 3, 2, 0 });
    defer logits.deinit();

    var best = try logits.argmax(&ctx, .expert); // i64 indices, no grad
    defer best.deinit();
    try std.testing.expectEqualSlices(i64, &.{1}, try best.dataConst());

    var top = try logits.topK(&ctx, .expert, 2, .k);
    defer top.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 3, 2 }, try top.values.dataConst());

    var selected: [2]usize = undefined;
    var weights: [2]f32 = undefined;
    try logits.routerTopK(&ctx, .expert, 2, .{}, &selected, &weights);
    try std.testing.expectEqual(@as(usize, 1), selected[0]);
    // normalize_selected renormalizes the top-k softmax mass to 1
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), weights[0], 1e-6);

    // multinomial: seed-deterministic categorical draws; zero-weight
    // classes are never selected.
    var probs = try fucina.Tensor(.{ .row, .class }).fromSlice(&ctx, .{ 1, 3 }, &.{ 0, 2, 0 });
    defer probs.deinit();
    var picks = try probs.multinomial(&ctx, .class, .sample, 3, 42, true);
    defer picks.deinit();
    try std.testing.expectEqualSlices(i64, &.{ 1, 1, 1 }, try picks.dataConst());
}
```

## 4.17 Indexing, assembly, and functional updates (`src/ag/tensor.zig`)

These produce owned tensors, differentiable in every tensor operand unless
noted; gradients route exactly through the index maps (gather scatters-adds,
slices narrow, pads drop the border). All materialize copies except
`narrow`, `select`, `slice` on step-1 ranges, `unbindInto`'s entries, and
`repeatAxis` with `n == 1`, which alias the source storage (see their
bullets).

- `gather(ctx, tag, indices: []const usize, out_tag)` — select rows along
  `tag`; the axis is retagged `out_tag` (which may equal `tag`). This IS
  torch.index_select with host-side indices (indices live host-side by
  design — ARCHITECTURE.md); `indexSelect` below is the tensor-valued
  spelling and lowers to this. For per-ELEMENT index tensors use
  `takeAlongAxis` below.
- `indexSelect(ctx, tag, indices, out_tag)` — torch.index_select: `gather`
  with a rank-1 **i64** index tensor (the argmax/topK/sort index
  convention — their outputs feed it directly; any other dtype is a
  compile error), read host-side into the same `[]usize` path. Entries
  outside `[0, dim(tag))` error with `IndexOutOfBounds` (negatives are not
  wrapped); duplicate indices accumulate their gradients (the scatter-add
  adjoint); the index tensor is control data outside the graph and can be
  released after the call.
- `getRows(ctx, tag, indices, out_tag)` — **block-quantized tensors only**:
  fused gather + dequantize of rows from a rank-2 weight into an f32 tensor
  (the embedding-lookup path; [§10](10-quantization.md)). No-grad (quantized tensors are
  constants).
- `narrow(ctx, tag, start, length)` — contiguous sub-range as a zero-copy
  **view**: it retains the source buffer and aliases its memory, so a
  mutation of the source through `data()` is visible through the result.
- `select(ctx, tag, index: isize)` — torch.select / `x[i]`: one position
  of `tag` with the axis removed (one strided view over the source with a
  single `StridedViewBackward` record — a zero-copy view). Negative
  `index` counts from the end; out of range errors with
  `IndexOutOfBounds`. The gradient is the exact scatter — unselected
  positions receive zero.
- `slice(ctx, spec)` — multi-axis basic slicing (torch/numpy
  `x[1:-1, ::2]`, positive steps): `spec` names the tags to slice, each
  field a `fucina.SliceRange`-shaped range — [§3.7](03-tensors-types-construction-and-data-access.md#37-views-and-structural-ops-srcagtensorzig-srctag_opszig-srcexecgather_scatterzig) has the full bounds
  contract (negatives from the end, `end = null` = dim, clamping, no
  negative steps — compose `flip`). Composed per-axis `narrow`/`sliceStep`
  in tag order; scope-required under gradients when more than one axis is
  sliced.
- `pad(ctx, tag, before, after, fill)` — constant padding on one axis; pad
  positions hold `fill` and drop their gradient.
- `concat(ctx, tag, others: []const *const Self)` — concatenation along an
  existing tag; one multi-parent node, differentiable in all inputs.
- `stack(ctx, new_tag, axis_index, others)` — stack along a **new** axis
  (composed insertAxis + concat; scope-required under gradients, [§4.1](04-tensor-operations.md#41-the-common-operation-contract-srcagtensorzig)).
- `unbindInto(ctx, tag, out: []Tensor(removeTag(tags, tag)))` — fill a
  caller-provided slice with the `dim(tag)` slices, each with `tag` removed
  (composed narrow + squeeze; scope-required under gradients). The caller
  owns and deinits every filled entry; on error, already-filled entries have
  been released.
- `repeatAxis(ctx, tag, n)` — tile the axis n times (`n == 1` is a
  zero-copy identity view; `n == 0` is `InvalidShape`); gradients from all
  copies accumulate.
- `flip(ctx, tag)` / `roll(ctx, tag, shift)` — reverse / rotate one axis
  (gather with a permutation; exact gradient).
- `setSlice(ctx, tag, start, update)` / `setRows(ctx, tag, indices, update)`
  — functional overwrite of a range / of specific rows; differentiable in
  both `self` (gradient zeroed where overwritten) and `update`.
- `zeroSlice(ctx, tag, start, length)` / `zeroRows(ctx, tag, indices)` —
  copy with a range/rows zeroed; the zeroed positions receive zero gradient.
- `maskedSelect(ctx, mask, out_tag)` — `torch.masked_select`: rank-1 tensor
  of the elements where `mask` is nonzero, row-major. The mask must be
  input-shaped, contiguous, and non-grad; `self` is differentiable (composed
  flatten + gather, scope-required under gradients). Selecting nothing is
  the dedicated `EmptySelection` (zero-size tensors are not representable),
  distinct from the shape errors so the no-match case is recoverable with a
  targeted `catch`; pre-counting with a mask sum avoids the error path
  (see the guard snippet in [§3.7](03-tensors-types-construction-and-data-access.md#37-views-and-structural-ops-srcagtensorzig-srctag_opszig-srcexecgather_scatterzig)).
- `indexAdd(ctx, tag, indices: []const usize, update)` —
  torch.index_add: `setRows`'s accumulating sibling — duplicates allowed,
  each occurrence adds. Differentiable in both (`self` identity, `update`
  row-gather).
- `takeAlongAxis(ctx, tag, indices)` — torch.gather /
  np.take_along_axis: per-element row selection along `tag`. `indices` is
  a same-tagged i64 tensor (the argmax/topK/sort index convention —
  their outputs feed it directly; any other dtype is a compile error),
  matching `self` on every other axis; the result takes `indices`' shape. Parallel over outer
  slices (disjoint writes — bitwise identical for any thread count);
  differentiable in `self` (exact scatter-add adjoint, duplicate reads
  accumulate).
- `scatterAdd(ctx, tag, indices, src)` / `scatter(ctx, tag, indices,
  src)` — torch.scatter_add / torch.scatter: functional per-element
  accumulate/overwrite at `indices` along `tag` (`indices` shaped exactly
  like `src`). Duplicates accumulate in `scatterAdd`; in `scatter` they
  resolve deterministically to the LAST row-major write (torch leaves the
  order unspecified; this pins it). Parallel over outer slices —
  duplicates only collide within a slice, where serial row-major order is
  preserved, so accumulation order and last-write-wins stay bitwise
  identical for any thread count. Differentiable in both operands
  (overwrite zeroes `self`'s gradient at every written slot; on
  duplicates every writer receives the winning slot's gradient — the
  torch formula).
- `nonzero(ctx)` — host-side `[]usize` of row-major nonzero flat
  indices ([§3.7](03-tensors-types-construction-and-data-access.md#37-views-and-structural-ops-srcagtensorzig-srctag_opszig-srcexecgather_scatterzig)); pairs with `gather`/`indexAdd`/`oneHot`.

```zig
test "takeAlongAxis pairs with argsort indices" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = fucina.Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 2, 3 }, &.{ 30, 10, 20, 5, 15, 0 });
    defer x.deinit();
    // argsort emits i64 indices — takeAlongAxis consumes them directly,
    // reordering each row (here: torch.gather(x, 1, x.argsort(1))).
    var order = try x.argsort(&ctx, .col, false);
    defer order.deinit();
    var sorted = try x.takeAlongAxis(&ctx, .col, &order);
    defer sorted.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30, 0, 5, 15 }, try sorted.dataConst());
}
```

```zig
test "gather, flip, and narrow copy with exact gradients" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{.d}).fromSlice(&ctx, .{3}, &.{ 10, 20, 30 });
    defer x.deinit();
    var picked = try x.gather(&ctx, .d, &.{ 2, 0 }, .g); // tag .d becomes .g
    defer picked.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 30, 10 }, try picked.dataConst());

    var reversed = try x.flip(&ctx, .d);
    defer reversed.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 30, 20, 10 }, try reversed.dataConst());

    var mid = try x.narrow(&ctx, .d, 1, 2);
    defer mid.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 20, 30 }, try mid.dataConst());
}
```

```zig
test "select, multi-axis slice, and tensor-valued indexSelect" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const M = fucina.Tensor(.{ .row, .col });
    var x = try M.fromSlice(&ctx, .{ 3, 4 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    defer x.deinit();

    var last_row = try x.select(&ctx, .row, -1); // Tensor(.{ .col }), torch x[-1]
    defer last_row.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 9, 10, 11, 12 }, try last_row.dataConst());

    // torch x[1:, 1:-1] — omitted bounds default, negatives count from the end.
    var inner = try x.slice(&ctx, .{ .row = .{ .start = 1 }, .col = .{ .start = 1, .end = -1 } });
    defer inner.deinit();
    var inner_mat = try inner.materialize(&ctx);
    defer inner_mat.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 6, 7, 10, 11 }, try inner_mat.dataConst());

    // index_select with an i64 tensor — argmax/topK/sort outputs feed it directly.
    const I = fucina.Tensor(.{ .dtype = .i64, .tags = .{.pick} });
    var idx = try I.fromSlice(&ctx, .{2}, &.{ 2, 0 });
    defer idx.deinit();
    var rows = try x.indexSelect(&ctx, .row, &idx, .pick); // Tensor(.{ .pick, .col })
    defer rows.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 9, 10, 11, 12, 1, 2, 3, 4 }, try rows.dataConst());
}
```

## 4.18 MoE facade entries (`src/exec/moe.zig`, `src/exec.zig`)

Routed expert FFNs run below the tag facade, directly on `ExecContext`
(inference-only; [§10](10-quantization.md) covers the quantized layouts, [§13](13-the-model-stack-fucina_models.md) the LLM integration):

```zig
pub const MoeRhs = union(enum) { q4_k: ..., q5_k: ..., q6_k: ..., q8_0: ...,
    tq2_0: ..., ptqtp: ..., q2_k: ..., iq2_xxs: ..., iq3_xxs: ..., iq2_s: ...,
    iq4_xs: ..., q3_k: ..., streamed: ... };  // fucina.MoeRhs
pub fn moeExpertFfn(self: *ExecContext, x: *const Tensor,
    gate: *const MoeRhs, up: *const MoeRhs, down: *const MoeRhs,
    selected: []const usize, weights: []const f32,
    out_pe: usize, act: GatedOp, io: ?std.Io, profile: ?*MoeBatchProfile) !Tensor
pub fn moeExpertFfnBatch(..., top_k: usize, ...) !Tensor
```

`fucina.MoeRhs` stacks all experts of one layer's gate/up/down projection
into a single compact-block RHS (experts are row-contiguous zero-copy
sub-views; the resident arms cover q4_k/q5_k/q6_k/q8_0/tq2_0/q2_k/
iq2_xxs/iq3_xxs/iq2_s/iq4_xs/q3_k, plus a `streamed` arm whose expert blocks resolve
through the disk-backed expert store (`src/store/expert_store.zig`)
instead of one resident buffer). The `ptqtp` arm holds K ∈ {1..3} tq2_0
plane stacks (PTQTP experts, [§10.9](10-quantization.md#109-ptqtp-multi-plane-ternary-decomposition-srcptqtpzig-ptqtpmd)): the fused op runs the ternary tile
once per plane and sums per element in fixed plane order before the gated
nonlinearity — bitwise the dense fused PTQTP linear on the same weights.
Tie-fitted K = 2 stacks ([§10.9](10-quantization.md#109-ptqtp-multi-plane-ternary-decomposition-srcptqtpzig-ptqtpmd), `--ptqtp-tie`) additionally carry the
4-bit folded pack (`MoePtqtpRhs.folded`, expert-major); when present, the
expert dot runs the one-pass folded kernel instead of two plane passes.
The streamed tier gathers a `ProjSpec` with `plane_count`/`plane_offsets`
pointing at the persisted `<name>.ptqtpK` sibling tensors ([§13.2.1](13-the-model-stack-fucina_models.md#132-weight-loading-srcweightszig)), and
`ProjSpec.fold` folds the two plane row-blocks into the pack on the way
into the slab (disk layout unchanged).
`moeExpertFfn` computes the route-weighted sum over the selected experts of
`down(act(gate(x), up(x)))` for a single token; `moeExpertFfnBatch` is the
batched-prefill variant taking the per-token `selected`/`weights` produced
by `routerTopK` ([§4.16](04-tensor-operations.md#416-selection-argmax-topk-sort-routertopk-srcagtensorzig-srcexectopkzig)). `fucina.MoeBatchProfile` is an optional wall-clock
breakdown the caller can pass to profile a run.

`ExecContext.moe_chain` re-exports the shared batched-MoE scheduling
scaffolding of `src/exec/moe_chain.zig` — the expert-grouped route plan,
the phase-chain machinery, chunk helpers, and the profile timer pair — so
in-tree LLM-band MoE engines ([§13](13-the-model-stack-fucina_models.md)) reach the exact same types through the
`fucina` root.

`fucina.expert_store` (`src/store/expert_store.zig`) is the out-of-core
expert tier behind the `streamed` arm: `fucina.ExpertStore`
(`create`/`destroy`, `addLayer`/`finalize`) opens the GGUF part files and
resolves expert blocks through a pinned → LRU → `pread` hierarchy inside
a caller-driven `acquire`/`release` window, so MoE models larger than RAM
decode against the same kernels as the resident arms. Demand-miss reads
fan out across a persistent I/O worker pool (`Options.io_workers`, default
8; the acquiring thread drains the same batch; `0` = sequential on the
calling thread), and `addMirror` attaches additional full copies of the
split GGUF before `finalize` — typically on other drives, with a per-copy
weight setting its read share relative to the primary's 1 — so expert
reads split across every copy. `Options.uncached` opens the store and
mirror fds uncached (macOS `F_NOCACHE`; no-op on Linux): a streamed giant
reads more than RAM per token, so page-caching those reads only evicts
the mmapped dense weights. `streamedRhs` hands out the per-projection
`StreamedMoeRhs` handle the `streamed` arm wraps; `pilotHint` enqueues
router-lookahead readahead on a dedicated I/O thread; `cacheRouteTopK` —
active only when `Options.cache_route: ?CacheRouteOptions` was set — is
opt-in cache-aware top-k selection (max-rank): the true top-`sacred`
ranks are always taken and the remaining slots prefer already-resident
experts among the top-`window` ranks (it changes routing, so callers opt
in explicitly, and `Stats` reports the swap rate); `saveUsage` persists
the expert-usage histogram to a `<gguf>.experts` sidecar that auto-pin
turns into a pinned hot tier at the next startup, and `repinPass` adapts
that tier live at generation boundaries.

## 4.19 Math on non-f32 tensors (`src/ag/tensor.zig`)

`Tensor(.{ .dtype = dt, .tags = ... })` instantiates typed constant tensors
([§3](03-tensors-types-construction-and-data-access.md), [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)). Their math surface is forward-only, always no-grad:

- **f16 / bf16 / f64** (`supportsForwardFloatMath`) — native typed kernels:
  `to` (cast), `add`, `sub`, `mul`, `div` (same dtype both sides — cast
  explicitly), `sum`, `mean`, `sumAll`, `dot`, `scale`, `divScalar`, plus
  the shared view set of every scalar-dtype branch ([§3.10](03-tensors-types-construction-and-data-access.md#310-facade-surface-index):
  the views, `gather`, `narrow`, `concat`, `setSlice`, `setRows`,
  `split`, `merge`, `flatten`, `reshape`, `flip`, `roll`, `stack`, ...).
  Pointwise and `dot` keep the input dtype;
  reductions widen f16/bf16 results to f32 ([§4.7](04-tensor-operations.md#47-reductions-and-scans-srcagtensorzig), [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)).
- **f16 / bf16 only — the ops with an f32 kernel.** These run widen → f32
  kernel → narrow-once (f32 arithmetic and accumulation with a single final
  round, the `widened` class of the [§8.3](08-data-types-storage-and-the-raw-tensor-layer-internal.md#83-float-computeoutput-dtype-policy-srcdtypezig) policy; on f64 they are
  a compile error — f64 math must not round through f32). The elementwise
  family is one mixin shared with the f32 branch (`src/ag/tensor/elementwise.zig`):
  its exec entries take the storage dtype and apply the policy themselves,
  so the 16-bit branches get the same methods with the same shapes and
  keep the input dtype: the unary family (`unary(op)` and every named
  alias listed in [§3.10](03-tensors-types-construction-and-data-access.md#310-facade-surface-index)), `leakyRelu`, `clamp`/`clampMin`/`clampMax`,
  `addScalar`, `subScalar`, `powScalar`, `log1p`, `maximum`, `minimum`,
  `gated`/`glu`/`swiglu`/`geglu`/`situ`, `splitGated`, `where`/`maskedFill`
  (`.bool` or float masks). The same holds for the shared softmax, scan,
  reduction, shape and norm mixins: `softmax` (plain `.{}` options; the ext
  options are the f32 kernel's), `logSoftmax`, `cumsum`, `cumprod`, `pad`,
  `rmsNorm`, `rmsNormMul`, `rmsNormMulAdd` (same-dtype weight and
  residual), `layerNorm` (plain, or `.weight`+`.bias` of the same dtype),
  the comparison family (`compare` with a `.bool` result, the logical
  combinators, `isnan`/`isinf`/`isfinite`), and `matmul`/`einsum`
  (same-dtype operands; the plain rank-2 `matmul` runs the typed GEMM, the
  other kinds, `bmm` and the `einsum` lowering widen; the result is the
  matmul output dtype). The reductions `max`, `min`,
  `prod`, `variance`, `logsumexp` return **f32** like the native typed
  `sum`/`mean` ([§8.3](08-data-types-storage-and-the-raw-tensor-layer-internal.md#83-float-computeoutput-dtype-policy-srcdtypezig)); `argmax` returns i64 ([§4.16](04-tensor-operations.md#416-selection-argmax-topk-sort-routertopk-srcagtensorzig-srcexectopkzig)).
- **Block-quantized** (q8_0, q4_k, ...): no arithmetic — `to(.f32)`
  (dequantize), `getRows` ([§4.17](04-tensor-operations.md#417-indexing-assembly-and-functional-updates-srcagtensorzig)), row-axis `concat`, `packRhs` /
  `packRhsAs` ([§4.9](04-tensor-operations.md#49-explicit-matmul-ternary-ste-and-packed-rhs-gemms-srcagtensorzig)), and constructors/views ([§3](03-tensors-types-construction-and-data-access.md), [§10](10-quantization.md)). Their main math
  role is as the constant RHS of `dot` ([§4.8](04-tensor-operations.md#48-dot-tag-directed-contraction-srcagtensorzig-srctag_opszig)) and `dotPacked` ([§4.9](04-tensor-operations.md#49-explicit-matmul-ternary-ste-and-packed-rhs-gemms-srcagtensorzig)).
- **Integer dtypes** (e.g. token-id tensors) — ordinary integer forward
  math, plain exec loops (integers are never the hot path): wrapping
  two's-complement `add`/`sub`/`mul` (torch's narrowing behavior),
  `maximum`/`minimum`, and EXPLICIT division — `divTrunc` (toward zero)
  and `divFloor` (toward −inf), `error.DivisionByZero` on a zero divisor,
  minInt/−1 wrapping to minInt. There is deliberately no integer `div`:
  torch's `/` silently promotes integers to float, and Fucina keeps
  promotion explicit (documented divergence). Remainders pair the
  divisions: `rem` pairs `divTrunc` (sign of the dividend, C `%` / Zig
  `@rem`) and `mod` pairs `divFloor` (sign of the divisor, Python/numpy
  `%` / Zig `@mod` — the n-gram-hash op); both error on a zero divisor
  and define minInt % −1 as 0. Bitwise combinators `bitAnd`/`bitOr`/
  `bitXor` operate on the two's-complement bit patterns (distinct from
  the `.bool` truthiness `logicalAnd/Or/Xor`; `bitXor` with a wrapping
  `mul` is exactly the multiplicative-XOR hash family, e.g. Engram [§13](13-the-model-stack-fucina_models.md)).
  All of these follow the standard tag-broadcast rule ([§4.2](04-tensor-operations.md#42-pointwise-binary-ops-and-tag-driven-broadcasting-srcagtensorzig-srctag_opszig)).
  `sum`/`sumAll` accumulate in i64 and RETURN `.i64` (torch's
  integer-sum dtype). `to` casts to any scalar dtype ([§3.8](03-tensors-types-construction-and-data-access.md#38-casting-todtype-srcagtensorzig-srcexecconvertzig)).
- **`.bool`**: no pointwise arithmetic (compile error — cast first);
  `to` and the counting `sum`/`sumAll` (i64) apply, plus the shared view
  set.

The typed forward ops are no-grad by design: an operand that requires
gradients is REJECTED with `error.UnsupportedGradient` instead of silently
dropping its graph. The differentiable ways into and out of the 16-bit
world are `to` ([§3.8](03-tensors-types-construction-and-data-access.md#38-casting-todtype-srcagtensorzig-srcexecconvertzig)) and the mixed-RHS `dot`/`einsum` ([§4.8](04-tensor-operations.md#48-dot-tag-directed-contraction-srcagtensorzig-srctag_opszig)); widen with
`to(.f32)` for everything else in a trained path.

Because the widened ops run the identical f32 kernels and round once on
store (in the exec entry), their
results are bit-identical to "cast up, run the f32 op, cast down" — pinned
by parity tests in `src/ag/tensor_tests/`:

```zig
test "bf16 forward ops compute through f32 and narrow once" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .seq, .d }).fromSlice(&ctx, .{ 2, 2 }, &.{ 0.5, -1.25, 2.0, 3.5 });
    defer x.deinit();
    var half = try x.to(&ctx, .bf16);
    defer half.deinit();

    var activated = try half.gelu(&ctx); // widen -> f32 gelu -> narrow
    defer activated.deinit();
    comptime std.debug.assert(@TypeOf(activated).dtype == .bf16);

    var reference_f32 = try x.gelu(&ctx);
    defer reference_f32.deinit();
    var reference = try reference_f32.to(&ctx, .bf16);
    defer reference.deinit();
    try std.testing.expectEqualSlices(fucina.Bf16, try reference.dataConst(), try activated.dataConst());

    // Widened reductions keep the f32 accumulator dtype, like sum/mean.
    var spread = try half.variance(&ctx, .d, .{ .ddof = 0 });
    defer spread.deinit();
    comptime std.debug.assert(@TypeOf(spread).dtype == .f32);
}
```

```zig
test "integer math wraps, divides explicitly, and reduces to i64" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const Ids = fucina.Tensor(.{ .dtype = .i8, .tags = .{.d} });
    var a = try Ids.fromSlice(&ctx, .{2}, &.{ 127, -7 });
    defer a.deinit();
    var b = try Ids.fromSlice(&ctx, .{2}, &.{ 1, 2 });
    defer b.deinit();

    var wrapped = try a.add(&ctx, &b); // two's-complement wrap
    defer wrapped.deinit();
    try std.testing.expectEqualSlices(i8, &.{ -128, -5 }, try wrapped.dataConst());

    var quotient = try a.divFloor(&ctx, &b); // explicit: no integer `div`
    defer quotient.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 127, -4 }, try quotient.dataConst());

    var remainder = try a.mod(&ctx, &b); // floored: pairs divFloor
    defer remainder.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 0, 1 }, try remainder.dataConst());

    var mixed = try a.bitXor(&ctx, &b); // two's-complement bit patterns
    defer mixed.deinit();
    try std.testing.expectEqualSlices(i8, &.{ 126, -5 }, try mixed.dataConst());

    var total = try a.sumAll(&ctx); // integer reductions return i64
    defer total.deinit();
    comptime std.debug.assert(@TypeOf(total).dtype == .i64);
    try std.testing.expectEqual(@as(i64, 120), try total.item());
}
```

The f32 `to(ctx, target_dtype)` cast is differentiable for `.f32 → .f32`
and for the mixed-precision narrows `.f32 → .f16`/`.bf16` ([§3.8](03-tensors-types-construction-and-data-access.md#38-casting-todtype-srcagtensorzig-srcexecconvertzig), [§5.1](05-automatic-differentiation.md#51-the-gradient-model-srcagtensorzig-srcagcorezig) —
the backward is the identity on the f32 upstream gradient); casting a
grad-requiring tensor to any other dtype fails with
`error.GradientCastUnsupported`.
