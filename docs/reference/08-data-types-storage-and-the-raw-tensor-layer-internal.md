# 8. Data types, storage, and the raw tensor layer (internal)

This section documents the substrate under the public `Tensor` facade: the
dtype system (`src/dtype.zig`), refcounted storage (`src/storage.zig`), and
the raw tensor value (`src/tensor.zig`). **None of this is a stable public
API.** The module root deliberately does not export the raw tensor type — a
comptime guard makes that a compile error ([§8.6](08-data-types-storage-and-the-raw-tensor-layer-internal.md#86-the-fucinainternal-escape-hatch-srcfucinazig)) — and the sanctioned
in-tree names for it are `fucina.internal.RawTensor` and, for
microbenchmarks only, `bench_raw.RawTensor` ([§2](02-toolchain-build-and-project-wiring.md)). It is documented here
because it is load-bearing for everything else: the dtype policy in [§8.3](08-data-types-storage-and-the-raw-tensor-layer-internal.md#83-float-computeoutput-dtype-policy-srcdtypezig)
explains every output dtype in [§4](04-tensor-operations.md), the buffer refcount explains the memory
model in [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md) and [MEMORY-MODEL.md](../MEMORY-MODEL.md), and anyone extending the
library (new ops, backend kernels, model loaders) works directly against
these types. Expect this surface to change without compatibility notice.

## 8.1 The `DType` enum (`src/dtype.zig`)

```zig
pub const DType = enum {
    bool, u8, u16, i8, i16, i32, i64, f16, bf16, f32, f64,
    f8_e4m3, f8_e5m2,
    q1_0, q2_0, q4_0, q4_1, q5_0, q5_1, q8_0, q8_1,
    q2_k, q3_k, q4_k, q5_k, q6_k, q8_k,
    iq1_s, iq1_m, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq4_nl, iq4_xs,
    tq1_0, tq2_0, mxfp4, nvfp4,
};

pub const DTypeKind = enum { scalar, block_quantized };
```

`DType` is the logical format tag carried by every buffer and tensor type. It
is re-exported at the public root as `fucina.DType`. Every dtype falls into
one of two kinds (`kind(dtype)`, `isScalar`, `isBlockQuantized`):

- **Scalar** dtypes store one storage element per logical tensor element.
- **Block-quantized** dtypes store one packed block struct per `blockSize`
  logical elements, always along the last logical axis.

Scalar dtypes:

| `DType` | `Scalar(dtype)` | Size | Notes |
|---|---|---|---|
| `.bool` | `bool` | 1 B | `zero()` is `false`, `one()` is `true` |
| `.u8` | `u8` | 1 B | |
| `.u16` | `u16` | 2 B | token-id workhorse |
| `.i8` | `i8` | 1 B | |
| `.i16` | `i16` | 2 B | |
| `.i32` | `i32` | 4 B | |
| `.i64` | `i64` | 8 B | |
| `.f16` | `f16` | 2 B | IEEE binary16, native Zig float |
| `.bf16` | `u16` | 2 B | **raw bfloat16 bits**, not a float type; `one(.bf16) == 0x3f80` |
| `.f32` | `f32` | 4 B | the only differentiable public dtype |
| `.f64` | `f64` | 8 B | |
| `.f8_e4m3` | `u8` | 1 B | **raw OCP FP8 E4M3FN bits** (NaN code, no infinities); storage-only float, `one(.f8_e4m3) == 0x38` |
| `.f8_e5m2` | `u8` | 1 B | **raw OCP FP8 E5M2 bits** (IEEE-like, ±inf); storage-only float, `one(.f8_e5m2) == 0x3c` |

Block-quantized dtypes (GGML-compatible wire formats; the packed `extern
struct` layouts are byte-exact against ggml and pinned by `comptime` size
asserts in `src/dtype.zig`). Encoding/decoding semantics per format are [§10](10-quantization.md);
this table is the storage geometry:

| `DType` | Block struct | Elems/block (`blockSize`) | Bytes/block (`blockByteSize`) |
|---|---|---|---|
| `.q1_0` | `BlockQ1_0` | 128 | 18 |
| `.q2_0` | `BlockQ2_0` | 128 | 34 |
| `.q4_0` | `BlockQ4_0` | 32 | 18 |
| `.q4_1` | `BlockQ4_1` | 32 | 20 |
| `.q5_0` | `BlockQ5_0` | 32 | 22 |
| `.q5_1` | `BlockQ5_1` | 32 | 24 |
| `.q8_0` | `BlockQ8_0` | 32 | 34 |
| `.q8_1` | `BlockQ8_1` | 32 | 36 |
| `.q2_k` | `BlockQ2_K` | 256 | 84 |
| `.q3_k` | `BlockQ3_K` | 256 | 110 |
| `.q4_k` | `BlockQ4_K` | 256 | 144 |
| `.q5_k` | `BlockQ5_K` | 256 | 176 |
| `.q6_k` | `BlockQ6_K` | 256 | 210 |
| `.q8_k` | `BlockQ8_K` | 256 | 292 |
| `.iq1_s` | `BlockIQ1_S` | 256 | 50 |
| `.iq1_m` | `BlockIQ1_M` | 256 | 56 |
| `.iq2_xxs` | `BlockIQ2_XXS` | 256 | 66 |
| `.iq2_xs` | `BlockIQ2_XS` | 256 | 74 |
| `.iq2_s` | `BlockIQ2_S` | 256 | 82 |
| `.iq3_xxs` | `BlockIQ3_XXS` | 256 | 98 |
| `.iq3_s` | `BlockIQ3_S` | 256 | 110 |
| `.iq4_nl` | `BlockIQ4_NL` | 32 | 18 |
| `.iq4_xs` | `BlockIQ4_XS` | 256 | 136 |
| `.tq1_0` | `BlockTQ1_0` | 256 | 54 |
| `.tq2_0` | `BlockTQ2_0` | 256 | 66 |
| `.mxfp4` | `BlockMXFP4` | 32 | 17 |
| `.nvfp4` | `BlockNVFP4` | 64 (16-elem subblocks) | 36 |

The block structs and the size constants (`q1_0_block_size`, `q2_0_block_size`,
`q4_0_block_size`, `q4_1_block_size`, `q5_0_block_size`, `q5_1_block_size`,
`q8_0_block_size`, `q8_1_block_size`, `qk_k_block_size` = 256,
`k_scale_size` = 12, `iq4_nl_block_size`, `mxfp4_block_size`,
`nvfp4_block_size`, `nvfp4_subblock_size`, `iq3s_n_scale`) are `pub` in
`src/dtype.zig`; the block structs are also re-exported at the public root
(`fucina.quant.BlockQ4_K`, `fucina.quant.BlockTQ2_0`, ...) because loaders and format
code legitimately handle raw blocks.

A block-quantized tensor of shape `[..., n]` requires `n` to be a nonzero
multiple of `blockSize(dtype)` and stores
`prefix_product * n / blockSize(dtype)` block structs
(`storageElementCount`, [§8.5.7](08-data-types-storage-and-the-raw-tensor-layer-internal.md#85-the-raw-tensor-tensorofdtype-srctensorzig)). Both `blockSize` and `blockByteSize` are
comptime functions; calling `blockSize` on a scalar dtype is a compile error.

## 8.2 Storage mapping and dtype predicates (`src/dtype.zig`)

```zig
pub fn Scalar(comptime dtype: DType) type       // scalar dtypes only
pub fn Storage(comptime dtype: DType) type      // any dtype
pub fn Accumulator(comptime dtype: DType) type  // scalar dtypes only
```

- `Scalar(dtype)` is the per-logical-element type. It is a compile error for
  block-quantized dtypes ("block-quantized dtypes do not have one scalar
  storage element per logical tensor element"). Note `Scalar(.bf16) == u16`:
  bf16 is stored and passed as raw bits everywhere.
- `Storage(dtype)` is the per-storage-element type: `Scalar(dtype)` for
  scalar dtypes, the block struct for block-quantized dtypes. Buffers and
  raw tensors are sized in `Storage(dtype)` units.
- `Accumulator(dtype)` is the reduction accumulator type: `f32` for
  `.f16`/`.bf16`/`.f32` and the f8 storage floats, `f64` for `.f64`, `u64`
  for `.bool`/`.u8`/`.u16`, `i64` for the signed integers. Compile error for
  block dtypes.

Classification predicates (all comptime, all `pub`):

| Function | True for |
|---|---|
| `kind(dtype)` | returns `.scalar` or `.block_quantized` |
| `isScalar` / `isBlockQuantized` | kind shorthands |
| `isFloat` | `.f16`, `.bf16`, `.f32`, `.f64` |
| `isF8` | `.f8_e4m3`, `.f8_e5m2` (OCP FP8 storage-only floats: convertible to/from f32, excluded from forward math and grads) |
| `isInteger` | `.u8`, `.u16`, `.i8`, `.i16`, `.i32`, `.i64` |
| `isSignedInteger` / `isUnsignedInteger` | the obvious subsets |
| `supportsGrad` | `== isFloat` (only float tensors can carry gradients; in practice only `.f32` does, [§5](05-automatic-differentiation.md)) |
| `supportsIntMath` | `== isInteger` (wrapping integer pointwise math and i64-accumulated reductions; `.bool` reduces but has no pointwise math) |
| `supportsForwardFloatMath` | `== isFloat` (forward-only math on the typed facade, [§3](03-tensors-types-construction-and-data-access.md)) |
| `supportsToFloat` | floats, the f8 storage floats, plus every block-quantized dtype (dequantizable) |
| `supportsQuantizedMatmulRhs` | every block dtype **except** `.q8_1` and `.q8_k` (those two are activation-side dot-product formats, [§10](10-quantization.md)) |
| `supportsQuantizedGetRows` | `== isBlockQuantized` (embedding-row gather) |
| `logicalDType` | blocks and the f8 storage floats map to `.f32`, other scalars map to themselves |

Scalar constant/conversion helpers: `zero(dtype)`, `one(dtype)`,
`name(dtype)` (the tag name), `toF32`/`toF64`/`fromF32`/`fromF64` (float
dtypes and the f8 storage floats; compile error otherwise), and
`castFloat(source_dtype, target_dtype, value)`, which routes through `f64`
when the target is `.f64` and through `f32` otherwise.
`castScalar(source_dtype, target_dtype, value)` is the general scalar cast
across the non-block dtypes: float↔float delegates to `castFloat`, the f8
storage floats bridge through `f32` in both directions (so their raw `u8`
storage never takes the integer legs), and the int/bool legs follow the
semantics quoted in [§3.8](03-tensors-types-construction-and-data-access.md#38-casting-todtype-srcagtensorzig-srcexecconvertzig)'s `to` conversion table (integer↔integer wraps
two's-complement, float→integer truncates toward zero and saturates with
NaN → 0, anything→bool is `!= 0`, bool→number is 0/1).
`isTruthy(dtype, value)` is mask truthiness: `!= 0`, with bf16 and the f8
formats read through the value bridge so `-0.0` stays falsy and NaN is
truthy. `toAccumulator`/`fromAccumulator` convert between `Scalar(dtype)`
and `Accumulator(dtype)` (bool maps to 0/1). `bf16ToF32(bits: u16) f32` and
`f32ToBf16(value: f32) u16` implement the bf16 bridge: round-to-nearest-even
on narrowing, with ggml-compatible NaN quieting (a NaN payload never
truncates to infinity; `src/dtype_tests.zig` pins this).

`f8e4m3ToF32`/`f32ToF8e4m3` and `f8e5m2ToF32`/`f32ToF8e5m2` implement the
OCP FP8 bridges. Decode is an exact comptime 256-entry table per format
(every representable value is exact in f32; NaN codes keep their sign).
Encode is round-to-nearest-even with per-format overflow semantics: e4m3
(the E4M3FN variant, no infinities) saturates to its NaN code like torch's
`float8_e4m3fn` cast, e5m2 overflows to ±inf; NaN inputs keep their sign
with the canonical NaN code. `src/dtype_tests.zig` pins all of this
exhaustively: every finite code round-trips bit-exactly, and every f16
value (which covers every rounding boundary of both formats) encodes
identically to a brute-force nearest-even search over the decode table.

## 8.3 Float compute/output dtype policy (`src/dtype.zig`)

```zig
pub const FloatOp = enum { pointwise, widened, reduction, matmul };

pub fn computeDType(comptime op: FloatOp, comptime input_dtype: DType) DType
pub fn outputDType(comptime op: FloatOp, comptime input_dtype: DType) DType
```

Forward float math has one explicit, comptime-resolved policy. `computeDType`
names the arithmetic/accumulation dtype, `outputDType` the result storage
dtype. For block-quantized dtypes both functions return the input unchanged;
integers and `.bool` follow the integer rows in the table:

| Op family | Input | Computes in | Returns |
|---|---|---|---|
| pointwise | `.f16` | `f16` | `.f16` |
| pointwise | `.bf16` | `f32` | `.bf16` |
| pointwise | `.f32` | `f32` | `.f32` |
| pointwise | `.f64` | `f64` | `.f64` |
| pointwise | integers | input dtype (wrapping) | input dtype |
| widened | `.f16`, `.bf16`, `.f32` | `f32` | input dtype |
| widened | `.f64` | `f64` (a compile error where no f64 kernel exists) | `.f64` |
| reduction | `.f16`, `.bf16`, `.f32` | `f32` | **`.f32`** |
| reduction | `.f64` | `f64` | `.f64` |
| reduction | integers, `.bool` | `i64` (wrapping) | **`.i64`** |
| dot/matmul | `.f16`, `.bf16`, `.f32` | `f32` (accumulate) | input dtype |
| dot/matmul | `.f64` | `f64` | `.f64` |

Three rules fall out of the table:

- **bf16 computes through f32 always** — it is stored as raw `u16` bits, so
  even pointwise ops widen via `bf16ToF32`, compute in `f32`, and narrow
  back with round-to-nearest-even on store (except reductions, which return
  `f32` outright).
- **`widened` is the class of the ops that have an f32 kernel only** (the
  unary family, `leakyRelu`, `clamp`, the scalar ops, `where`/`maskedFill`,
  `gated`/`splitGated`, float `max`/`min`, `compare`/`compareScalar`,
  `softmax`/`logSoftmax`/`logsumexp`, the scans,
  `prod`/`varAxis`/`argmax`/`maxAxis`/`minAxis`, `rmsNorm`/`layerNorm`,
  `bmm`, the non-plain `matmul` kinds and the `einsum` lowering). Their exec entries take the storage dtype and
  apply the policy themselves: `ctx.prepareAs(dtype, compute, x)` widens on
  entry, `ctx.storeAs(compute, dtype, value)` narrows once on store
  ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)); the reductions among them return the reduction output
  dtype (f32 for 16-bit inputs). f32 inputs pay nothing (both are the
  contiguous borrow and the value itself). `pad` is pure data movement and
  runs on the storage dtype directly.
- **Reductions on 16-bit floats return f32.** Summing `f16`/`bf16` into a
  16-bit result would lose the accumulator's precision, so the widened
  result dtype is kept.
- **Explicit casts are required for anything else.** No op silently promotes
  across operand dtypes: mixed-dtype pointwise math on the typed facade is a
  compile error (`"typed pointwise requires matching dtypes; cast
  explicitly"` in `src/ag/tensor.zig`). Casting is an explicit op — `to(ctx,
  target_dtype)` on the public facade ([§3](03-tensors-types-construction-and-data-access.md)), `cast` on `ExecContext`
  ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)).

The policy is visible directly in public result types:

```zig
test "float dtype policy: f16 reduction returns f32" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const Half = fucina.Tensor(.{ .dtype = .f16, .tags = .{ .row, .col } });
    var x = try Half.fromSlice(&ctx, .{ 2, 2 }, &.{ 1.5, 2.5, 3.0, 4.0 });
    defer x.deinit();

    // Pointwise keeps the input dtype: f16 + f16 -> f16.
    var y = try x.add(&ctx, &x);
    defer y.deinit();
    comptime std.debug.assert(@TypeOf(y).dtype == .f16);

    // Reductions on f16 accumulate in f32 and *return* f32.
    var s = try x.sum(&ctx, .col, .{});
    defer s.deinit();
    comptime std.debug.assert(@TypeOf(s).dtype == .f32);
    try std.testing.expectEqualSlices(f32, &.{ 4.0, 7.0 }, try s.dataConst());

    // Any other output dtype requires an explicit cast.
    var wide = try x.to(&ctx, .f32);
    defer wide.deinit();
    comptime std.debug.assert(@TypeOf(wide).dtype == .f32);
}
```

## 8.4 Refcounted storage: `BufferOf(dtype)` (`src/storage.zig`)

```zig
pub fn BufferOf(comptime buffer_dtype: DType) type {
    return struct {
        allocator: Allocator,
        data: []Elem,                     // Elem == dtype.Storage(buffer_dtype)
        refs: std.atomic.Value(u32),
        release_ctx: ?*anyopaque = null,
        release_fn: ?*const fn (*anyopaque, *Self) void = null,
        pending_work: std.atomic.Value(?*accelerator.Work) = .init(null),
        pending_use: std.atomic.Value(?*accelerator.Work) = .init(null),
        accelerator_resource: std.atomic.Value(?*accelerator.Resource) = .init(null),

        pub const dtype = buffer_dtype;
        pub const Element = Elem;
        ...
    };
}
pub const Buffer = BufferOf(.f32);
```

A buffer is a heap-allocated header (`allocator.create(Self)`) plus a typed
data slice, shared by pointer and lifetime-managed by an atomic refcount.
Every raw tensor holds exactly one reference to exactly one buffer; views
share the buffer by taking additional references.

Constructors (all return `!*Self` with `refs == 1`):

| Constructor | Data ownership | At `refs == 0` |
|---|---|---|
| `create(allocator, len)` | owned, uninitialized `len` elements | `destroy()`: free data + header |
| `createWithRelease(allocator, len, release_ctx, release_fn)` | owned | `release_fn(release_ctx, self)` |
| `fromSlice(allocator, values)` | owned copy of `values` | `destroy()` |
| `fromBorrowedSlice(allocator, values)` | **aliases** caller memory | destroy header only; caller keeps the bytes |
| `fromBorrowedSliceWithRelease(allocator, values, release_fn)` | aliases | `release_fn(self, self)` — full cleanup duty |
| `fromBorrowedSliceWithReleaseCtx(allocator, values, release_ctx, release_fn)` | aliases | `release_fn(release_ctx, self)` — full cleanup duty |

Refcount operations:

- `retain()` — `fetchAdd(1, .monotonic)`. Safe from any thread.
- `release()` — `fetchSub(1, .acq_rel)`; debug-asserts against
  over-release. When the count reaches zero it invokes `release_fn` if set,
  otherwise `destroy()`. Exactly one caller observes zero, so the hook fires
  **exactly once** (`src/storage_tests.zig` pins this).
- `isUnique()` — acquire-load snapshot `refs == 1`. **Snapshot only**: it is
  meaningful only when the caller already has exclusive access to the tensor
  handle pointing at this buffer (the basis of `canTakeInPlace`, [§8.5.5](08-data-types-storage-and-the-raw-tensor-layer-internal.md#85-the-raw-tensor-tensorofdtype-srctensorzig)).
- `resetRefs()` — stores 1; valid only under exclusive ownership. Used by the
  `ExecContext` buffer pool when recycling a cached buffer ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)).
- `destroy()` — unconditionally frees data + header, bypassing the refcount.
  Only for owners that know no references remain (pool teardown).
- `waitReady()` / `discardPending()` — complete already-submitted GPU output
  work, respectively making host bytes visible or skipping an unused D2H.
  `waitReady` is safe under CONCURRENT callers (parallel chunk copies land
  N readers on one buffer): a single claimant dereferences and releases the
  Work; everyone else spins until the slot clears, which happens only after
  the host copy is visible (8-thread regression in `src/storage_tests.zig`).
- `setPendingUse()` / `waitUnused()` / `waitMutable()` — track the latest
  submitted GPU reader of this allocation. Const host reads may overlap a
  device read; mutable access waits so post-call input mutation cannot race
  Metal zero-copy reads or CUDA async upload. Provider queue order lets the
  latest token subsume earlier readers. Final release always completes both
  output and reader work before storage can be recycled.
- `acceleratorResource` — provider mapping metadata tied to this allocation's
  lifetime (Metal's pooled page-wrapper cache; CUDA host page registration).
  It survives ordinary pool release/reacquire and is destroyed with the
  backing allocation.

**Release-hook contract.** For the borrowed-with-release variants the hook
runs once, at the final `release()`, and takes *full* cleanup responsibility:
it must dispose of the external data by whatever means created it **and**
free the header with `buffer.destroyHeader()` (which also releases any
accelerator resource). Capture the external slice, call `destroyHeader()`
first, then free/unmap the bytes: provider teardown may still need the live
address to unregister it. The two in-tree
production uses:

- **GPU device-resident bytes** — `src/weights.zig` wraps managed device
  allocations from `internal.gpu.allocResidentBytes` so the final buffer
  release (counting `cloneView`'d weights that share it) frees them via
  `internal.gpu.freeResidentBytes` ([§8.6](08-data-types-storage-and-the-raw-tensor-layer-internal.md#86-the-fucinainternal-escape-hatch-srcfucinazig), [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)).
- **pooled slabs** — `src/exec/buffer_pool.zig` uses the `Ctx` variant so a
  typed buffer's release returns its byte slab to the pool free list instead
  of freeing it ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)).

Note the tree does **not** use release hooks for mmap'd GGUF bytes: that
lifetime is holder-managed — `gguf.File.deinit` munmaps, or ownership moves
via `takeMapping` to a `MappedRegion` the holder must keep alive while
anything borrows tensor data from it ([§12](12-model-io-gguf-and-safetensors.md)). The hook mechanism remains the
right tool when *user* code wants refcount-driven cleanup of an external
mapping, as below:

```zig
fn wrapMappedWeights(alloc: std.mem.Allocator, mapped: []f32) !fucina.internal.RawTensor {
    const RawTensor = fucina.internal.RawTensor;
    const Buffer = std.meta.Child(@FieldType(RawTensor, "buffer")); // BufferOf(.f32)

    const hook = struct {
        fn releaseMapped(_: *anyopaque, buffer: *Buffer) void {
            // Full cleanup responsibility: the external bytes AND the header.
            const bytes = std.mem.sliceAsBytes(buffer.data);
            buffer.destroyHeader();
            std.posix.munmap(@alignCast(bytes));
        }
    };

    const buffer = try Buffer.fromBorrowedSliceWithRelease(alloc, mapped, hook.releaseMapped);
    return RawTensor.fromOwnedBuffer(buffer, &.{ 4, 8 }) catch |err| {
        buffer.release(); // still owns the one reference on failure
        return err;
    };
    // Later: the final tensor deinit drops refs to 0 and fires the hook once.
}
```

The buffer type is not separately exported; internal code names it through
the tensor's field type, as above (the `src/weights.zig` idiom).

**Thread-safety.** `retain`/`release` are atomic and may race freely; the
data slice is not synchronized — concurrent reads are fine, and writers need
external coordination (the runtime's parallel kernels partition disjoint
ranges, [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)). `isUnique` and `resetRefs` are only correct under exclusive
access as described above.

## 8.5 The raw tensor: `TensorOf(dtype)` (`src/tensor.zig`)

```zig
pub const max_rank = 8;

pub const TensorError = error{
    ShapeMismatch, InvalidShape, InvalidDataLength, IndexOutOfBounds, UnsupportedView,
    EmptySelection, DivisionByZero,
};

pub const Shape = struct {
    len: u8,
    dims: [max_rank]usize,
    pub fn init(values: []const usize) !Shape        // rejects rank 0/>8 and zero dims
    pub fn initStrides(values: []const usize) !Shape // zeros allowed (broadcast strides)
    pub fn slice(self: *const Shape) []const usize
    pub fn at(self: *const Shape, i: usize) usize
};

pub fn TensorOf(comptime tensor_dtype: DType) type {
    return struct {
        buffer: *BufferOf(tensor_dtype),
        shape: Shape,
        strides: Shape,
        offset: usize = 0,

        pub const dtype = tensor_dtype;   // and: pub const Element = Storage(dtype)
        ...
    };
}
pub const Tensor = TensorOf(.f32);        // == fucina.internal.RawTensor
```

A raw tensor is a plain value: a buffer pointer plus **inline** shape/stride
metadata (two fixed `[8]usize` arrays — no allocation per view) and a start
`offset`, all measured in *storage elements* (`Storage(dtype)` units — for
block-quantized dtypes, strides count blocks). Rank is runtime (1 to
`max_rank` = 8; there is no rank-0 shape, which is why the facade's
scalar-tag tensor is a rank-1 `{1}` raw tensor). Copying the struct does
**not** retain the buffer; every legitimately owned tensor value carries
exactly one buffer reference, and `deinit()` releases it and poisons the
struct (`self.* = undefined` — not idempotent).

The raw tensor appears inside public signatures (`ctx.fromSlice` returns
one; `Tensor(spec).variable(&ctx, raw)` consumes one), but the type itself is
only nameable as `fucina.internal.RawTensor` /
`fucina.internal.tensor_mod.TensorOf(dtype)`. For convenience `tensor.zig`
re-exports `DType`, `Scalar`, and `Storage` from `dtype.zig` (and
`storage.zig` re-exports `DType`), so `tensor_mod` alone is enough for most
raw-layer work.

### 8.5.1 Construction and ownership

| Constructor | Dtypes | Semantics |
|---|---|---|
| `zeros(allocator, shape)` / `ones(allocator, shape)` | scalar only (compile error otherwise) | fresh owned buffer, filled |
| `fromSlice(allocator, shape, values: []const Scalar)` | scalar only | owned copy; `InvalidDataLength` unless `values.len == elementCount(shape)` |
| `fromBorrowedSlice(allocator, shape, values: []Scalar)` | scalar only | aliases caller memory (borrowed buffer; caller keeps ownership of the bytes and must outlive the tensor) |
| `fromStorageSlice(allocator, shape, values: []const Element)` | any | owned copy in storage elements; for block dtypes `values.len` must equal `storageElementCount` |
| `fromBorrowedStorageSlice(allocator, shape, values: []Element)` | any | borrowed, in storage elements |
| `fromOwnedBuffer(buffer, shape)` | any | **consumes one reference** to `buffer`; the caller must not release that reference after success — `deinit` does. Accepts oversized buffers (`data.len >= storageElementCount`), which is how pooled buffers are wrapped; `InvalidDataLength` if too small. On error the reference stays with the caller |
| `scalar(allocator, value)` | scalar only | shape `{1}` |
| `clone(allocator)` | any | materializing deep copy into a fresh contiguous buffer (see 8.5.4) |

All errors are `TensorError` members, plus `error.Overflow` from checked
element-count multiplication and `error.OutOfMemory` from every allocating
constructor (only `fromOwnedBuffer` is allocation-free). The allocator
passed at construction is stored in the buffer and used for its teardown.

### 8.5.2 Geometry queries

`rank()`, `len()` (logical element count), `storageLen()` (storage element
count; `len()/blockSize` per trailing axis for block dtypes), `rows()` /
`cols()` (rank-2 only, else `InvalidShape`), `isScalar()` (`len() == 1`),
and `isContiguous()` — true when strides are exactly the row-major strides
of the shape (a broadcast `{1}` scalar with stride 0 is *not* contiguous).

### 8.5.3 Views

All view constructors `retain()` the buffer and return a new tensor value
that must be `deinit`ed independently; shape/stride metadata is copied
inline, and `offset` is preserved (or extended). Writes through any view are
visible through every alias of the same buffer.

```zig
pub fn cloneView(self: *const Self) !Self
pub fn reshape(self: *const Self, new_shape: []const usize) !Self
pub fn viewWithStrides(self: *const Self, shape: []const usize, strides: []const usize) !Self
pub fn viewWithStridesOffset(self: *const Self, shape: []const usize, strides: []const usize, offset_delta: usize) !Self
pub fn broadcastTo(self: *const Self, target_shape: []const usize) !Self
pub fn broadcastToRank(self: *const Self, comptime target_rank: usize, target_shape: [target_rank]usize) !Self
```

- `cloneView` — identical view, one more reference. This is how weight
  tensors are shared across module structs.
- `reshape` — **requires contiguity** (`UnsupportedView` otherwise) and a
  matching element count (`InvalidShape`); the result is a retained view
  over the same storage, never a copy. Non-contiguous tensors must be
  materialized first (`clone`, or `ExecContext.materialize*` / the tagged
  layer's `contiguousForReshape`, [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)/[§7](07-named-axes-the-tag-algebra.md)).
- `viewWithStrides` / `viewWithStridesOffset` — arbitrary strided
  (sub)views, **checked**: `strides.len` must match `shape.len`
  (`InvalidShape`) and the maximal reachable index
  `offset + offset_delta + Σ (dim-1)·stride` must lie inside the buffer
  (`InvalidDataLength`). `offset_delta` advances the view's start; this is
  the raw narrowing primitive (there is no dedicated `narrow` on the raw
  type — `ExecContext.narrowAxis` in [§6](06-the-execution-runtime-execcontext-and-the-memory-model.md) and the
  facade's `narrow` in [§4](04-tensor-operations.md) are built on it).
- `broadcastTo` / `broadcastToRank` — zero-stride broadcast views,
  right-aligned like NumPy: the source rank must not exceed the target rank
  (`ShapeMismatch`), new leading axes get stride 0, matching axes keep
  their stride, size-1 axes get stride 0, and any other mismatch is
  `ShapeMismatch`. `broadcastTo` dispatches over runtime target rank 1–8
  (`InvalidShape` beyond); `broadcastToRank` takes the rank at comptime.

**Block-quantized restriction:** for block dtypes, `reshape` and the
`viewWithStrides*` family accept only the identity view (same shape, same
strides, `offset_delta == 0`) and otherwise return `UnsupportedView`; blocks
are indivisible, so only whole-tensor aliasing is a view. Broadcasting
follows the generic path but is only meaningful on non-trailing axes.

```zig
fn rawViewTour(alloc: std.mem.Allocator, blocks: []const fucina.quant.BlockQ8_0) !void {
    const RawTensor = fucina.internal.RawTensor; // TensorOf(.f32)

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

    // Zero-stride broadcast view; reshape of a contiguous tensor is a view.
    var b = try x.broadcastTo(&.{ 4, 2, 3 });
    defer b.deinit();
    var flat = try x.reshape(&.{6});
    defer flat.deinit();

    // Non-f32 raw tensors: fucina.internal.tensor_mod.TensorOf(dtype).
    var ids = try fucina.internal.tensor_mod.TensorOf(.u16)
        .fromSlice(alloc, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer ids.deinit();

    // Block-quantized raw tensor: last axis in logical elements, storage in
    // blocks (here 4*64/32 == 8 BlockQ8_0 storage elements).
    var w = try fucina.internal.tensor_mod.TensorOf(.q8_0)
        .fromStorageSlice(alloc, &.{ 4, 64 }, blocks);
    defer w.deinit();
}
```

### 8.5.4 Data access and materialization

```zig
pub fn data(self: *Self) []Elem                    // PANICS on non-contiguous
pub fn dataConst(self: *const Self) []const Elem   // PANICS on non-contiguous
pub fn dataChecked(self: *Self) ![]Elem            // error.UnsupportedView instead
pub fn dataConstChecked(self: *const Self) ![]const Elem
pub fn item(self: *const Self) Elem                // scalar dtypes; asserts len() == 1
pub fn copyTo(self: *const Self, dst: []Elem) !void
pub fn clone(self: *const Self, allocator: Allocator) !Self
```

`data`/`dataConst` return `buffer.data[offset .. offset + storageLen()]` and
**panic** on non-contiguous tensors (`"Tensor.data requires a contiguous
tensor; materialize or use dataChecked"`) — they are for hot paths that have
already established contiguity. `dataChecked`/`dataConstChecked` are the
recoverable variants (`UnsupportedView`). `item()` debug-asserts a
single-element tensor and reads through `dataConst` (so it also requires
contiguity — a zero-stride broadcast scalar panics).

`copyTo(dst)` writes the logical contents into a caller slice of exactly
`storageLen()` elements (`InvalidDataLength` otherwise): a straight `memcpy`
when contiguous, an odometer copy for non-contiguous scalar tensors (the
maximal row-major-contiguous axis suffix moves as whole `memcpy` runs, a
strided innermost axis as a stride-increment loop — never a per-element
division; dim-1 axes are absorbed, so a spuriously non-contiguous singleton
view still copies as one `memcpy`), and `UnsupportedView` for non-contiguous
block tensors. `copyRangeTo(dst, linear_start, count)` is the range form
(scalar dtypes): disjoint ranges of the row-major linearization may be
copied concurrently, which is how the exec runtime parallelizes large
materializations ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)). `clone(allocator)` is the materialization path: it
allocates a fresh contiguous buffer and `copyTo`s into it — the result is
always contiguous with `offset == 0`, regardless of the source view.

### 8.5.5 In-place helpers and `canTakeInPlace`

Scalar-dtype-only mutators (compile error for block dtypes):
`addInPlace(other)` (`ShapeMismatch` unless shapes match; both operands go
through `data()`/`dataConst()` and hence panic when non-contiguous),
`scaleInPlace(scalar_value)`, and `fill(value)`.

```zig
// Safe only when the caller owns exclusive access to this Tensor value.
// The refcount proves no other retained Tensor aliases the buffer now; it
// is not a lock against another thread retaining the same Tensor later.
pub fn canTakeInPlace(self: *const Self) bool  // offset == 0 and isContiguous() and buffer.isUnique()
```

`canTakeInPlace` is the **ownership optimization** used by consuming ops to
steal an operand's buffer and write the result in place instead of
allocating. The exact contract: it returns true only for a full-buffer
(`offset == 0`), contiguous, uniquely-referenced tensor, and the answer is
trustworthy **only while the caller has exclusive access to the tensor
handle** — the refcount check proves no *other* view aliases the buffer at
that instant; it is *not* synchronization, and another thread that could
still retain/read the same handle invalidates the optimization by contract,
not by any runtime check.

### 8.5.6 Fixed-rank views

```zig
pub fn RankedTensorOf(comptime tensor_dtype: DType, comptime rank: usize) type {
    return struct {
        tensor: *const TensorOf(tensor_dtype),
        shape: [rank]usize,
        strides: [rank]usize,
        pub fn dim(self: @This(), comptime axis: usize) usize
        pub fn len(self: @This()) usize
        pub fn isContiguous(self: @This()) bool
    };
}
pub fn RankedTensor(comptime rank: usize) type  // f32 alias

pub fn rankView(self: *const Self, comptime rank_value: usize) !RankedTensorOf(dtype, rank_value)
```

`rankView` copies the runtime shape/strides into comptime-sized arrays
(`InvalidShape` when the tensor's rank differs). The result **borrows** the
tensor — no retain; it must not outlive it. This is the bridge from
runtime-rank tensors to rank-specialized kernels: with `[rank]usize` in
hand, loops unroll at comptime (`inline while`/`inline for`), which is how
the exec layer's `*Rank` entry points ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)) and backend kernels ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)) are
written.

### 8.5.7 Shape arithmetic (free functions)

`pub` helpers in `src/tensor.zig`, shared by the exec and tagged layers:

- `requireSameShape(a, b)` / `requireSameShapeOf(dtype, a, b)` —
  `ShapeMismatch` unless shapes are equal.
- `elementCount(shape)` / `elementCountArray(rank, shape)` — logical element
  count; `InvalidShape` for rank 0/>8 or zero dims; overflow-checked.
- `storageElementCount(dtype, shape)` / `storageElementCountArray(...)` —
  storage element count; for block dtypes the last axis must be a nonzero
  multiple of `blockSize(dtype)` (`InvalidShape` otherwise).
- `elementCountArrayAssumeValid(rank, shape)` — unchecked product for
  already-validated shapes.

**Thread-safety.** Raw tensors have no interior locking. The only atomic
state is the buffer refcount; concurrent readers of one buffer are safe,
concurrent writers (or a writer racing readers) need external coordination.
The runtime never mutates shared storage concurrently except by partitioning
disjoint ranges across the worker team ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)).

## 8.6 The `fucina.internal` escape hatch (`src/fucina.zig`)

The public root deliberately does **not** export the raw tensor type, and an
anti-regression guard makes reintroducing it a compile error on any build
that analyzes the module root (every test, example, and tool — not just
`zig build test`):

```zig
comptime {
    if (@hasDecl(@This(), "RawTensor")) @compileError(
        "fucina.RawTensor must not be exported at the public root; raw tensors are internal. " ++
            "Use fucina.internal.RawTensor (in-tree raw naming) or bench_raw.RawTensor (microbench).",
    );
}
```

The rationale is API shape, not capability: the no-grad `Tensor` facade has
negligible forward overhead, so model and example code carries
`fucina.Tensor(spec)` end-to-end, and a public raw type would split the
ecosystem into two tensor vocabularies. Code that genuinely needs the raw
layer names it through `fucina.internal`; raw-kernel microbenchmarks use the
separate `bench_raw` module (`src/bench_raw.zig`), which the guard does not
affect (it inspects only the root's own decls).

```zig
pub const internal = struct {
    pub const backend_mod = backend;      // src/backend.zig
    pub const tensor_mod = tensor;        // src/tensor.zig
    pub const thread_mod = thread;        // src/thread.zig
    pub const gpu = struct { ... };       // GPU hooks, see below
    pub const RawTensor = tensor.Tensor;  // TensorOf(.f32)
};
```

- `backend_mod`, `tensor_mod`, `thread_mod` — the internal surface for
  sibling modules (notably `fucina_models`, [§13](13-the-model-stack-fucina_models.md)) that need **exact core type
  identity** without importing a second copy of the backend/exec files: a
  `TensorOf(.q4_k)` from a re-imported `tensor.zig` would be a distinct,
  incompatible type. `tensor_mod` gives typed raw tensors
  (`TensorOf(dtype)`, `RankedTensorOf`, the shape helpers); `thread_mod` the
  thread-pool primitives (`Pool`, `WaitGroup`, `Mutex`, ..., [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md));
  `backend_mod` the kernel entry points and packed-RHS types ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)).
- `RawTensor` — the canonical internal name for the raw no-grad f32 tensor.
  Intended users: runtime/backend internals, raw-kernel benchmarks,
  serialization/format byte work, and tests targeting raw runtime behavior.
- `gpu` — hooks for model loaders and benchmark instrumentation,
  deliberately kept off the public root: users keep ordinary eager `Tensor`
  values; residency and tracing are backend-owned details ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)).

| Hook | Type / signature | Purpose |
|---|---|---|
| `enabled` | `bool` (comptime) | true on GPU builds (`-Dgpu=metal` or `-Dgpu=cuda`, [§2](02-toolchain-build-and-project-wiring.md)): GPU GEMM offload is compiled in |
| `has_quant_gemm` | `bool` (comptime) | provider implements dequant-in-kernel quantized GEMM (dense + grouped MoE). Loaders that reshape CPU representations for the GPU quant path key on this, **not** on `enabled` — a provider can be enabled while its quantized arms are still CPU-only |
| `has_q5_k_quant` | `bool` (comptime) | provider additionally implements Q5_K dense/grouped quantized kernels (CUDA only at present, [§9.9.2](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#99-gpu-offload-srcbackendgpuzig-metalzig-cudazig)) |
| `has_tq2_0_quant` | `bool` (comptime) | provider additionally implements the TQ2_0 dequant-in-kernel GEMM (Metal only at present, [§9.9.1](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#99-gpu-offload-srcbackendgpuzig-metalzig-cudazig)); the `.tq2_0` GPU offload arms and the ternary/PTQTP loader layouts key on it |
| `allocResidentBytes` | `fn (len: usize) ?[]u8` | device-owned bytes for GPU-build loaders; `null` when unavailable (no device context, `len == 0`, or too large) |
| `freeResidentBytes` | `fn (bytes: []const u8) void` | release bytes returned by `allocResidentBytes`; safe no-op when the device context is gone or the slice is foreign |
| `traceEnabled` | `fn () bool` | opt-in dispatch tracing, enabled by `FUCINA_GPU_TRACE=1` |
| `traceReset` | `fn () void` | reset trace counters (call before a warm measurement window); no-op when tracing is off |
| `traceDump` | `fn () void` | print the accumulated dispatch/time breakdown to stderr; no-op when tracing is off |

The hooks resolve at comptime through `src/backend/gpu.zig` to the active
provider (`src/backend/metal.zig` or `src/backend/cuda.zig`; `-Dgpu=none`
resolves to the null provider `src/backend/gpu_none.zig`, with
`enabled == false` and every capability false, so the default build
analyzes neither real provider). Call sites gate
shim-touching calls on `comptime gpu.enabled` so CPU-only builds
comptime-elide every provider reference; `traceReset`/`traceDump` may be
called unconditionally on GPU builds since they no-op when tracing is off.

```zig
fn residentScratch(len: usize) !void {
    const gpu = fucina.internal.gpu;
    if (comptime gpu.enabled) {
        // Device-owned bytes for GPU-build loaders; null when unavailable.
        const bytes = gpu.allocResidentBytes(len) orelse return error.OutOfMemory;
        defer gpu.freeResidentBytes(bytes);

        // Dispatch tracing (FUCINA_GPU_TRACE=1); reset/dump no-op when off.
        gpu.traceReset();
        if (gpu.traceEnabled()) gpu.traceDump();
    }
}
```

For inspection (as opposed to construction), the public facade already
crosses the boundary: every public tensor exposes `asRawTensor()`, returning
`*const` raw tensor whose metadata can be read without owning anything:

```zig
test "asRawTensor exposes raw shape/stride metadata" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var x = try fucina.Tensor(.{ .rows, .cols }).fromSlice(&ctx, .{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x.deinit();

    const raw = x.asRawTensor(); // *const fucina.internal.RawTensor
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, raw.shape.slice());
    try std.testing.expectEqualSlices(usize, &.{ 3, 1 }, raw.strides.slice());
    try std.testing.expectEqual(@as(usize, 0), raw.offset);
    try std.testing.expect(raw.isContiguous());

    // Scalar-tag tensors are rank-1 shape {1} at the raw layer (no rank 0).
    var total = try x.sumAll(&ctx);
    defer total.deinit();
    try std.testing.expectEqualSlices(usize, &.{1}, total.asRawTensor().shape.slice());
}
```
