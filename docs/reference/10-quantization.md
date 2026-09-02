# 10. Quantization

Fucina treats ggml/llama.cpp block-quantized weights as first-class tensor
data: every GGUF quantized wire format is a `DType` whose storage element is
the exact ggml block struct, decoders and encoders are byte-for-byte ports of
`ggml-quants.c`, and matmuls against quantized weights run int8 dot-product
kernels over dynamically quantized activations. The stack has three tiers:

1. **dtype/block tier** (`src/dtype.zig`, [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)) — block structs, block-size
   constants, and the comptime predicates (`isBlockQuantized`,
   `supportsQuantizedMatmulRhs`, `supportsQuantizedGetRows`, `blockSize`,
   `logicalDType`).
2. **kernel tier** (`src/backend/quant.zig` and `src/backend/quant/*.zig`) —
   encoders, decoders, activation quantizers, dot/matmul kernels, packed RHS
   layouts, and the format-trait table. Reachable in-tree as
   `fucina.internal.backend_mod.quantized_matmul`, which forwards the block
   and RHS types and exposes the format modules by name
   (`quantized_matmul.q4_k.X`, `.q8k.X`, `.cold.X`); application code
   normally never calls it directly.
3. **facade tier** (`src/ag/tensor.zig`, `src/exec/quant_matmul.zig`) —
   `fucina.Tensor(.{ .dtype = .q4_k, ... })` constant tensors, `dot` with a
   quantized RHS, `getRows`, `to(.f32)`, `packRhs`/`dotPacked`, and the
   ternary STE training op `dotTernarySte`.

GGUF container mechanics (parsing, `tensorByteLen`, zero-copy tensor bytes)
are [§12](12-model-io-gguf-and-safetensors.md); the LLM weight wrappers that own packed RHS containers are [§13](13-the-model-stack-fucina_models.md); the
CPU/GPU dispatch machinery underneath is [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md).

## 10.1 Format inventory (`src/backend/quant/types.zig`, `src/dtype.zig`)

Each quantized `DType` stores rows as a contiguous sequence of fixed-size
blocks; a row of `k` logical elements is `k / block_size` blocks (`k` must
divide exactly — `QuantizedFormatError.InvalidQuantizedLength` otherwise).
The block structs are `extern struct`s matching ggml's wire layout exactly
and are re-exported at the root (`fucina.quant.BlockQ4_K`, ...). The formats are
registered once in `dtype.block_formats` (dtype tag, block struct,
elems/block); `Storage`, `kind`, `blockSize`, and GGUF's type mapping all
derive from that table, and a comptime completeness check makes a `DType`
tag claimed by neither the registry nor the scalar list a compile error —
adding a format is one block struct, one enum tag, and one registry row.

| DType | Block struct | Elems/block | Bytes/block | f32 encoder | Matmul kernel | LHS activation |
|---|---|---|---|---|---|---|
| `.q1_0` | `BlockQ1_0` | 128 | 18 | — | cold | Q8_0 |
| `.q2_0` | `BlockQ2_0` | 128 | 34 | yes | **hot** (mul-free ternary) | Q8_0 |
| `.q4_0` | `BlockQ4_0` | 32 | 18 | yes | cold | Q8_0 |
| `.q4_1` | `BlockQ4_1` | 32 | 20 | yes | cold | Q8_1 |
| `.q5_0` | `BlockQ5_0` | 32 | 22 | yes | cold | Q8_0 |
| `.q5_1` | `BlockQ5_1` | 32 | 24 | yes | cold | Q8_1 |
| `.q8_0` | `BlockQ8_0` | 32 | 34 | yes | **hot** (+ x4 packed) | Q8_0 |
| `.q8_1` | `BlockQ8_1` | 32 | 36 | yes | — (activation format) | — |
| `.q2_k` | `BlockQ2_K` | 256 | 84 | — | cold | Q8_K |
| `.q3_k` | `BlockQ3_K` | 256 | 110 | — | cold | Q8_K |
| `.q4_k` | `BlockQ4_K` | 256 | 144 | yes | **hot** (+ x8 / x2mmla packed) | Q8_K |
| `.q5_k` | `BlockQ5_K` | 256 | 176 | yes | **hot** (+ x8 packed) | Q8_K |
| `.q6_k` | `BlockQ6_K` | 256 | 210 | yes | **hot** (+ x4 packed) | Q8_K |
| `.q8_k` | `BlockQ8_K` | 256 | 292 | yes | — (activation format) | — |
| `.iq1_s` | `BlockIQ1_S` | 256 | 50 | — | cold | Q8_K |
| `.iq1_m` | `BlockIQ1_M` | 256 | 56 | — | cold | Q8_K |
| `.iq2_xxs` | `BlockIQ2_XXS` | 256 | 66 | — | cold | Q8_K |
| `.iq2_xs` | `BlockIQ2_XS` | 256 | 74 | — | cold | Q8_K |
| `.iq2_s` | `BlockIQ2_S` | 256 | 82 | — | cold | Q8_K |
| `.iq3_xxs` | `BlockIQ3_XXS` | 256 | 98 | — | cold | Q8_K |
| `.iq3_s` | `BlockIQ3_S` | 256 | 110 | — | cold | Q8_K |
| `.iq4_nl` | `BlockIQ4_NL` | 32 | 18 | — | cold | Q8_0 |
| `.iq4_xs` | `BlockIQ4_XS` | 256 | 136 | — | cold | Q8_K |
| `.tq1_0` | `BlockTQ1_0` | 256 | 54 | — | cold | Q8_K |
| `.tq2_0` | `BlockTQ2_0` | 256 | 66 | yes | **hot** (mul-free ternary) | Q8_K |
| `.mxfp4` | `BlockMXFP4` | 32 | 17 | — | cold | Q8_0 |
| `.nvfp4` | `BlockNVFP4` | 64 | 36 | — | cold | Q8_0 |

Reading the table:

- **f32 encoder** — the format has a `quantizeRowForDType` prong ([§10.6](10-quantization.md#106-encoders-ggufencodef32-and-ggml-parity-srcbackendquantzig-srcggufzig)).
  Every format decodes to f32 (`dequantizeRowForDType` covers all 27), so
  "encoder = —" means *decode/matmul-only*: usable as loaded weights, never
  producible in-process or by `gguf.encodeF32`.
- **Matmul kernel** — *hot* formats have dedicated SIMD kernels plus packed
  (column-interleaved) RHS layouts tuned per ISA; *cold* formats
  (`src/backend/quant/cold.zig`) have a generic per-block dot path that is
  correct and tested but not benchmark-tuned. `.q8_1` and `.q8_k` have no
  matmul kernel at all (`supports_matmul == false`): they exist as the
  *activation* side of the int8 dots and as decodable tensor data.
- **LHS activation** — the block format the f32 activations are dynamically
  quantized to before the int8 dot ([§10.5](10-quantization.md#105-lhs-activation-quantization-srcbackendquantq8kzig-srcbackendnativezig)): Q8_0 for the 32-element weight
  families plus `.q1_0`/`.q2_0` (128) and `.nvfp4` (64), Q8_1 for the offset formats
  `.q4_1`/`.q5_1` (the offset term needs the per-block activation sum Q8_1
  carries), Q8_K for all 256-element formats.
- First-class end-to-end (encoder + hot kernel + GGUF export): `.q8_0`,
  `.q4_k`, `.q5_k`, `.q6_k`, `.tq2_0`, `.q2_0` — the first four additionally
  have facade packed (column-interleaved) RHS layouts; the ternary formats
  `.tq2_0`/`.q2_0` have no facade pack (`PackedRhs(dt)` has no ternary arm;
  `packRhs` on a `.tq2_0` or `.q2_0` tensor is a compile error),
  though the kernel tier ships an x4 column-interleaved TQ2_0 pack
  (`packMatmulRhsTQ2_0x4`, [§10.7](10-quantization.md#107-ternary-tq2_0-first-class-tq1_0-decode-only-srcbackendquantternaryzig-ternarymd)) that the PTQTP weight wrappers consume
  ([§13.2.1](13-the-model-stack-fucina_models.md#132-weight-loading-srcweightszig)).

`DType` is the one identity of every format; `dtype.block_formats`
describes each block format programmatically (block struct, logical
elements per block), read through `dtype.blockSize`, `dtype.blockByteSize`,
`dtype.Storage`, `dtype.isBlockQuantized`, and
`dtype.supportsQuantizedMatmulRhs` (false for `q8_1`/`q8_k`, the two block
formats without an RHS kernel). Every RHS container (`QuantizedMatmulRhsQ4_K`,
the packs, `QuantizedRowsFor(dt)`, ...) carries `pub const dtype: DType`, and
`AnyQuantizedMatmulRhs` is tagged by `DType` names. Block-size constants
(`q8_0_block_size` = 32, `qk_k_block_size` = 256, `k_scale_size` = 12,
`iq4_nl_block_size`/`mxfp4_block_size` = 32, `nvfp4_block_size` = 64,
`nvfp4_subblock_size` = 16, `q1_0_block_size`/`q2_0_block_size` = 128, `q4_0_block_size`,
`q4_1_block_size`, `q5_0_block_size`, `q5_1_block_size`,
`q8_1_block_size`) originate in `src/dtype.zig`; `fucina.quant.q8_0_block_size`
is re-exported at the root.

Separate from the ggml formats, the kernel tier also ships a Fucina-native
**W8A8** container (`QuantizedMatmulRhsI8`, `AnyQuantizedMatmulRhs` tag
`.fucina_w8a8_rhs`): symmetric per-(column, group) int8 weights stored
transposed `[n][k]` with f32 scales (`default_group_size` = 32, resolved by
`effectiveGroupSize`/`groupCountForSize`; it carries no `DType`), built by
`quantizeRhsBlockwiseI8`
and multiplied by `matmulI8BlockwiseTile`/`matmulI8BlockwiseRange` against
per-row int8 activations (`quantizeActivationsPerRowI8`). It is not a tensor
dtype and has no facade surface; it exists for W8A8 experiments below the
facade.

## 10.2 The block-quantized public tensor (`src/ag/tensor.zig`)

`fucina.Tensor(.{ .dtype = <quant dtype>, .tags = ... })` instantiates a
*quantized constant tensor*: a tagged wrapper over the typed raw tensor whose
element type is the block struct. It deliberately exposes **only** quantized
operations — no autograd (`requiresGrad()` is always `false`, there is no
`variable`), no float math (`add`, `softmax`, ... are absent at comptime).
The public surface:

```zig
// constructors (all validate rank/shape; shapes are LOGICAL element counts)
pub fn constant(ctx: *ExecContext, value: RawTypedTensor) !Self          // consumes value on success
pub fn fromTensor(ctx: *ExecContext, value: RawTypedTensor) !Self        // alias of constant
pub fn fromBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []const Elem) !Self   // copies blocks
pub fn fromStorageSlice(...) !Self                                       // alias of fromBlocks
pub fn fromBorrowedBlocks(ctx: *ExecContext, raw_shape: [tensor_rank]usize, values: []Elem) !Self // zero-copy borrow

// accessors / structure
pub fn deinit(self: *Self) void
pub fn data(self: *Self) ![]Elem
pub fn dataConst(self: *const Self) ![]const Elem
pub fn copyTo(self: *const Self, dst: []Elem) !void
pub fn asRawTensor(self: *const Self) *const RawTypedTensor
pub fn axis(comptime tag: Tag) usize
pub fn hasTag(comptime tag: Tag) bool
pub fn dim(self: *const Self, comptime tag: Tag) usize
pub fn shape(self: *const Self) [tensor_rank]usize
pub fn requiresGrad(_: *const Self) bool                                 // always false
pub fn withTags(self, ctx, comptime new_tags_spec) !Tensor(...)          // retag view, same rank

// quantized operations
pub fn to(self, ctx, comptime target_dtype: DType) !Tensor(...)          // .f32 only: dequantize
pub fn materialize(self, ctx) !Self                                      // copy into owned storage
pub fn concat(self, ctx, comptime tag, others: []const *const Self) !Self// rank-2, row axis only
pub fn getRows(self, ctx, comptime tag, indices: []const usize, comptime out_tag) !Tensor(f32 ...)
pub fn packRhs(self, ctx) !PackedRhs(dtype)                              // §10.3
pub fn packRhsAs(self, ctx, comptime Rhs: type) !Rhs                  // explicit container
```

Semantics:

- **Shapes are logical.** A `[n, k]` quantized tensor holds
  `n * (k / block_size)` blocks; `fromBlocks` fails with
  `TensorError.InvalidDataLength` when the slice length disagrees and with
  `TensorError.InvalidShape` when `k` is not a multiple of the block size
  (the shape check fires before the kernel tier's
  `QuantizedFormatError.InvalidQuantizedLength` is ever reached on this
  path). Weight tensors follow the ggml convention
  `[out, in]` = `[n, k]`: row `r` of blocks is *output column* `r`.
- **Ownership.** `fromBlocks` copies into context-owned storage;
  `fromBorrowedBlocks` borrows — the caller keeps ownership and the blocks
  must outlive the tensor (this is the mmap'd-GGUF path: the loader
  reinterprets mapped tensor bytes as a block slice and borrows them
  zero-copy; see [§12](12-model-io-gguf-and-safetensors.md)). `materialize` converts a borrowed tensor into an
  owned copy. `constant`/`fromTensor` consume the raw tensor on success;
  on error, ownership stays with the caller. Every constructor's result is
  released with `deinit`.
- **`to(.f32)`** dequantizes the whole tensor (rows × columns) into a fresh
  f32 tensor with the same tags; any target other than `.f32` is a compile
  error. **`getRows`** gathers rows from the *first* axis by index and
  dequantizes only those rows — the embedding-lookup path (token embeddings
  stay quantized; only the looked-up ids are widened). Errors:
  `TensorError.InvalidShape` for empty `indices`,
  `TensorError.IndexOutOfBounds` for an index ≥ row count. `getRows` is
  comptime-restricted to rank-2 tensors; `to(.f32)` requires a rank-2
  value at runtime (`TensorError.InvalidShape` otherwise).
- **`concat`** joins quantized tensors along the row axis without
  dequantizing (rank-2, axis 0 only — comptime-checked).
- **Thread-safety.** Quantized tensors are immutable after construction;
  concurrent reads are safe. All ops take an `ExecContext`, which is
  single-threaded externally ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)) — internal parallelism comes from the
  context's work pool.

An asset-free construction path exists because the first-class encoders run
in-process: `fucina.gguf.encodeF32` ([§10.6](10-quantization.md#106-encoders-ggufencodef32-and-ggml-parity-srcbackendquantzig-srcggufzig)) turns f32 data into wire blocks,
which `fromBlocks` accepts directly:

```zig
test "encode f32 rows to Q8_0 blocks and dequantize them back" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // Two 32-element rows -> one BlockQ8_0 per row.
    var src: [2 * fucina.quant.q8_0_block_size]f32 = undefined;
    for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) * 0.25;

    var blocks: [2]fucina.quant.BlockQ8_0 = undefined;
    try fucina.gguf.encodeF32(.q8_0, &src, std.mem.sliceAsBytes(&blocks));

    const W = fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
    var w = try W.fromBlocks(&ctx, .{ 2, fucina.quant.q8_0_block_size }, &blocks);
    defer w.deinit();

    var dense = try w.to(&ctx, .f32); // block-wise dequantize
    defer dense.deinit();
    for (try dense.dataConst(), src) |got, want|
        try std.testing.expectApproxEqAbs(want, got, 0.01);
}
```

### f32 × quantized-RHS `dot`

The f32 tensor's `dot` ([§4](04-tensor-operations.md)) dispatches on the RHS dtype at comptime: when the
RHS is a block-quantized tensor whose dtype satisfies
`supportsQuantizedMatmulRhs` (everything in the table except `.q8_1` and
`.q8_k`), the contraction runs the quantized matmul instead of a dense GEMM.
Requirements, all comptime-checked:

- the RHS must have exactly one free axis and be stored **[free, contract]**
  (e.g. weight tags `{ .out, .in }` contracted over `.in`) — the ggml weight
  layout, never transposed at runtime;
- no shared batch tags between LHS and RHS;
- the LHS may have any number of free axes (they are flattened to `m` rows
  around the contraction).

The runtime path quantizes the LHS activations ([§10.5](10-quantization.md#105-lhs-activation-quantization-srcbackendquantq8kzig-srcbackendnativezig)), runs the format's
kernel, and reshapes back. Gradients: the quantized weight is a constant
(it never receives grad); the LHS gradient is supported and flows through
the **dequantized** weight — the backward node holds a view of the block
data and dequantizes it transiently (`ConstRhsDotBackward`,
`src/ag/backward/matmul.zig`). On GPU builds the forward may offload to the
dense-quant GEMM provider (q4_k/q6_k/q8_0 on Metal, plus q5_k on CUDA, [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md))
with or without gradients: the LHS gradient never reads the forward
kernel's internals (it flows through the dequantized weight), so the GPU
and CPU forwards share one backward and differ only by the
activation-quantization gap (the CPU kernels quantize the LHS, [§10.5](10-quantization.md#105-lhs-activation-quantization-srcbackendquantq8kzig-srcbackendnativezig);
measured 3.7e-3 max rel at a 512x1024x2048 q8_0 shape, gradient
bit-identical). Checkpoint blocks pin both of their runs to the CPU
kernels ([§5.5](05-automatic-differentiation.md#55-activation-checkpointing-srcagcheckpointzig)). Each backend's dispatch gate prices its own memory
system — Metal by work size alone (unified memory), CUDA with
residency-aware work floors — so the policy needs no per-backend carve-out.

```zig
test "f32 activations contract against a Q8_0 weight tensor" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    var wsrc: [2 * fucina.quant.q8_0_block_size]f32 = undefined;
    for (wsrc[0..fucina.quant.q8_0_block_size]) |*v| v.* = 1.0;
    for (wsrc[fucina.quant.q8_0_block_size..]) |*v| v.* = 2.0;
    var blocks: [2]fucina.quant.BlockQ8_0 = undefined;
    try fucina.gguf.encodeF32(.q8_0, &wsrc, std.mem.sliceAsBytes(&blocks));

    const W = fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
    var w = try W.fromBlocks(&ctx, .{ 2, fucina.quant.q8_0_block_size }, &blocks);
    defer w.deinit();

    const x_values = [_]f32{1} ** fucina.quant.q8_0_block_size;
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 1, fucina.quant.q8_0_block_size }, &x_values);
    defer x.deinit();

    var y = try x.dot(&ctx, &w, .in); // y: .{ .batch, .out }
    defer y.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 32), (try y.dataConst())[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 64), (try y.dataConst())[1], 1e-2);

    var row = try w.getRows(&ctx, .out, &.{1}, .seq); // f32 gather of weight row 1
    defer row.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 2), (try row.dataConst())[0], 1e-2);
}
```

## 10.3 RHS containers and packed layouts (`src/backend/quant/types.zig`, `src/exec/quant_matmul.zig`)

Two families of weight-side containers exist below the tensor facade, both
re-exported at the root:

**Plain per-format containers** — `fucina.quant.QuantizedMatmulRhsQ2_K`,
`fucina.quant.QuantizedMatmulRhsQ4_K`, `fucina.quant.QuantizedMatmulRhsQ5_K`,
`fucina.quant.QuantizedMatmulRhsQ6_K` (root); the kernel tier additionally
defines `QuantizedMatmulRhsQ8_0`, `QuantizedMatmulRhsQ4_0`,
`QuantizedMatmulRhsQ3_K`, the
`QuantizedMatmulRhsRowsFor(dtype)` generic (instantiated as
`QuantizedMatmulRhs{Q1_0,Q2_0,Q4_1,Q5_0,Q5_1,IQ1_S,IQ1_M,IQ2_XXS,IQ2_XS,IQ2_S,
IQ3_XXS,IQ3_S,IQ4_NL,IQ4_XS,TQ1_0,TQ2_0,MXFP4,NVFP4}`), the
row wrappers (`QuantizedRowsQ8_1` instantiates the `QuantizedRowsFor(dtype)`
generic; `QuantizedRowsQ8_0` and `QuantizedRowsQ4_0` are hand-written, and
`QuantizedRowsQ4_0`'s allocator is non-optional — no borrow support), and
the type-erased `AnyQuantizedMatmulRhs` union the backends dispatch on. A
plain container is blocks + `k`/`n` dims; the hot-format containers (`Q8_0`,
`Q4_K`, `Q5_K`, `Q6_K`), the `Q2_K`/`Q3_K` containers, and the
`QuantizedRowsFor` generic carry an
**optional allocator**:
`allocator = null` means the blocks are *borrowed* (mmap'd GGUF, packed ES
genomes) and `deinit` frees nothing. Ordinary users never build these —
`ExecContext` wraps tensor blocks in stack-allocated borrow containers per
dispatch, and the LLM MoE loader borrows expert blocks through
`fucina.MoeRhs` ([§13](13-the-model-stack-fucina_models.md)).

**Packed containers** — independent load-time snapshots consumed by
`dotPacked`. Dense packs are owned f32 output-row panels
(`src/backend/packed.zig`); quantized packs are column-interleaved copies laid
out so the innermost loop feeds the target's int8 dot instruction
(`sdot`/`smmla` on aarch64, VNNI/AVX2 on x86). Root exports:

```zig
pub fn PackedRhs(comptime dt: DType) type   // dense panel or ISA-best quantized layout
pub const QuantizedMatmulRhsQ8_0x4;         // + Q4_Kx4, Q4_Kx8, Q4_Kx2Mmla, Q5_Kx8, Q6_Kx4
pub const supports_q4_k_mmla: bool;         // aarch64 + i8mm target feature
```

`PackedRhs(dt)` (the backend's `PackedRhsFor`) maps f32/f16/bf16 to the
`PackedDenseRhs` f32 panel, q8_0→x4, q6_k→x4, q5_k→x8, and q4_k→x2mmla on
aarch64+i8mm targets, x8 otherwise. Each container owns its snapshot
(`PackedDenseRhs.rhs` or quantized blocks), holds `k`/`n`, and carries a
comptime `dtype` that `dotPacked` dispatches on (the Q4_K x8/x2mmla split is
the container type itself: compare `@TypeOf(rhs)` against
`fucina.quant.QuantizedMatmulRhsQ4_Kx2Mmla`). The `Q4_Kx4` container exists
only for kernel comparisons and has no facade entry (comptime error).

Packing and consuming happen on the facade:

- Dense `w.packRhs(ctx)` — snapshot a rank-2 contiguous f32/f16/bf16
  `[out, contract]` tensor as a `PackedDenseRhs` panel; f16/bf16 widen once. Logical
  output rows stay contiguous and the row count is padded to the four-output
  microkernel tile. Equivalent `ExecContext` entry:
  `packDenseMatmulRhs`.
- Quantized `w.packRhs(ctx)` / `w.packRhsAs(ctx, Rhs)` — pack a
  rank-2 contiguous tensor (`TensorError.UnsupportedView` when not
  contiguous). `packRhsAs` is the escape hatch to force a non-default
  container (e.g. `QuantizedMatmulRhsQ4_Kx8` on MMLA hardware to exercise
  the fused kernels). Equivalent
  `ExecContext` entries: `packMatmulRhs(dt, &w)` (the dtype-default
  container) and `packMatmulRhsAs(Rhs, &w)` (a specific container).
- Every pack is independent of the source tensor, which may be released after
  packing. The owner calls `deinit()` on the packed container.
- `x.dotPacked(ctx, &packed, contract_tag, out_tag)` — rank-2 f32 LHS stored
  `[free, contract]`; returns `[free, out_tag]`. **No gradient support**:
  dense packs return `error.GradientPackedMatmulUnsupported` and quantized
  packs return `error.GradientQuantizedMatmulUnsupported` when `self`
  requires grad.
- `x.rmsNormMulDotPacked(ctx, &norm_weight, eps, &packed, contract_tag, out_tag)`
  — fused pre-norm + packed GEMM: normalizes up to 4 rows at a time into
  task-private scratch with the exact `rmsNormMulRows` kernel and quantizes
  with the fused packers, so the normalized tensor is never materialized;
  matches `rmsNormMul` + `dotPacked` to ≤ 1 ulp observed (the packed
  matmul's internal LHS quantizer arrangement may differ in the last ulp,
  the `splitSwiGluDotPacked` precedent). Kernels exist for
  `q8_0x4`/`q4_kx8`/`q5_kx8`/`q6_kx4`; `q4_kx2mmla` is a deliberate
  comptime error — callers fall back to the unfused pair.
- `gate_up.splitSwiGluDotPacked(ctx, &packed, split_tag, out_tag)` — fused
  split-SwiGLU activation + down-projection GEMM without materializing the
  gated tensor; kernels exist for `q8_0x4`/`q4_kx8`/`q5_kx8`/`q6_kx4`
  (`q4_kx2mmla` is a deliberate comptime error — callers guard with
  `comptime !fucina.quant.supports_q4_k_mmla` and fall back to unfused).
- `gate.gegluQuantDotPacked(ctx, &up, &packed, in_tag, out_tag)` — fused
  GeGLU + down projection, `q8_0x4` only.

On `ExecContext`, every quantized matmul entry is a spelling of one
request pair (`src/exec/quant_matmul.zig`):

```zig
pub const QuantMatmul = struct {
    prologue: ?FusedActKind = null,    // .split_swiglu / .rms_norm_mul / .geglu_quant
    placement: enum { auto, cpu } = .auto,
    rhs_lifetime: RhsLifetime = .transient,
    numerics: enum { batched, rowwise } = .batched,
};
pub const Lhs = union(enum) { plain, rms_norm, gate_up };  // the prologue's operands
```

`matmulQuant(ctx, lhs, rhs, opts)` returns the f32 `[m, rhs.n]` product of
the (optionally fused-activated) LHS rows against a packed or compact RHS
container; `matmulQuantInto(ctx, out, lhs, rhs, opts)` is the
caller-storage form. `.plain` carries plain rows (the fused gate|up rows,
width 2k, for `.split_swiglu`); `.rms_norm` is `{ x, weight, eps }`;
`.gate_up` is `{ gate, up }` (`.geglu_quant`, `q8_0x4` only).
`placement = .auto` consults the GPU offload seam exactly as the named
entries did, falling back to the CPU kernels on decline (a lane-packed
container's `raw` blocks, set by the weight loader, serve the attempt);
an open `pinRowwiseNumerics` scope pins every row to the m == 1 kernels
context-wide, the K-quant fused engine by forcing its per-row tail kernel
instead of looping. RHS containers
come from `packMatmulRhs`/`packMatmulRhsAs` (owned lane packs),
`packDenseMatmulRhs` (the dense f32 panel), or the borrowing
`compactMatmulRhs`/`compactMatmulRhsFromBlocks` (a compact `.rows`
container over a block-quantized tensor's blocks or a raw block slice,
any `supportsQuantizedMatmulRhs` dtype; the container borrows the blocks
and needs no deinit).

One seam sits below it: the backend addresses its quantized kernels by an
`ops.QuantGemm` request, `{ weight: DType, rhs: RhsPack{rows, x4, x8,
x2mmla}, lhs: LhsForm{f32, q8_k, q8_kx4, q8_kx2mmla, q8_0, q8_0x4, q8_1},
order: LoopOrder{row_outer, col_outer} }`, whose `supported()` is the
one matrix of existing kernels (an unsupported combination is a compile
error naming it). Every packed container states its interleave as
`pub const pack`, each format file exports one
`gemm(comptime g, out, lhs, rhs, tile)` over its tile bodies, and
`quant.gemm` dispatches on `g.weight`.

At the LLM layer ([§13](13-the-model-stack-fucina_models.md)), `fucina_models`'s `weights.zig` wraps each quantized
projection as a struct holding the original blocks plus a
`fucina.PackedRhs(dtype)` built once at load. Inkling uses the same pattern
for its dense vision-tower projections and dense unembed: pack once while
loading, `dotPacked` per step, deinitialize with the owning model.

```zig
test "packed RHS matmul matches the unpacked quantized dot" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const m = 3;
    const n = 8;
    const k = 2 * fucina.quant.q8_0_block_size;

    var wsrc: [n * k]f32 = undefined;
    for (&wsrc, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 13 + 5) % 17)) * 0.1 - 0.8;
    var blocks: [n * 2]fucina.quant.BlockQ8_0 = undefined;
    try fucina.gguf.encodeF32(.q8_0, &wsrc, std.mem.sliceAsBytes(&blocks));

    const W = fucina.Tensor(.{ .dtype = .q8_0, .tags = .{ .out, .in } });
    var w = try W.fromBlocks(&ctx, .{ n, k }, &blocks);
    defer w.deinit();
    var packed_rhs = try w.packRhs(&ctx); // fucina.PackedRhs(.q8_0)
    defer packed_rhs.deinit();

    var xv: [m * k]f32 = undefined;
    for (&xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 7 + 3) % 11)) * 0.2 - 1.0;
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ m, k }, &xv);
    defer x.deinit();

    var fast = try x.dotPacked(&ctx, &packed_rhs, .in, .out);
    defer fast.deinit();
    var reference = try x.dot(&ctx, &w, .in);
    defer reference.deinit();
    for (try fast.dataConst(), try reference.dataConst()) |a, b|
        try std.testing.expectApproxEqAbs(b, a, 5e-2);
}
```

## 10.4 `RhsLifetime` and the address-keyed caching rule (`src/exec/quant_matmul.zig`)

The enum itself, the address-keyed cacheability rule, and the
pooled-storage prohibition are [§6.7](06-the-execution-runtime-execcontext-and-the-memory-model.md#67-rhslifetime-address-keyed-caching-of-rhs-operands-srcexecquant_matmulzig); this subsection covers how the flag
rides through quantized matmul.

The lifetime rides on the `QuantMatmul` request
(`matmulQuant`/`matmulQuantInto`, [§10.3](10-quantization.md#103-rhs-containers-and-packed-layouts-srcbackendquanttypeszig-srcexecquant_matmulzig)) and on the `try*` GPU attempts.
The facade `dot` always uses the default
transient/GPU-when-not-training request — a `.transient` RHS may
still use the provider's blocking GPU path, but no address-keyed wrap survives
the call and the borrowed bytes cannot be retained by an async command.
`fucina_models`'s weight wrappers thread `.stable_process` through for resident
or mmap'd weights ([§13](13-the-model-stack-fucina_models.md)); that lifetime first tries the direct-output async
path (Metal admits up to 8192 activation rows per dense-quant submission;
CUDA grows its slot table) and falls back to balanced blocking chunks of at most 2048 rows
when necessary. Q4_K/Q5_K/Q6_K/Q8_0 prefill uses provider- and format-specific work
gates calibrated against the actual compact/raw or load-time-packed CPU
fallback. Decode `m <= 8` remains behind the provider's explicit GEMV opt-in.
The complete GPU contract is [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md).

## 10.5 LHS activation quantization (`src/backend/quant/q8k.zig`, `src/backend/native.zig`)

Quantized matmuls quantize the f32 activations on the fly, per weight-format
family (all ggml-parity):

- **Q8_0** (`quantizeRowQ8_0Into`, `quantizeRowsQ8_0Into`,
  `quantizeRowsQ8_0`): per-32 symmetric absmax — `d = amax/127` stored as
  f16, `q = round(x/d)` clamped to i8. Consumed by q1_0/q2_0/q4_0/q5_0/q8_0 and
  the table formats iq4_nl/mxfp4/nvfp4.
- **Q8_1** (`quantizeRowQ8_1Into`, `quantizeRowsQ8_1`, defined in
  `src/backend/quant/cold.zig`): Q8_0 plus
  the per-block activation sum (`s = d·Σq` as f16) that the offset formats
  q4_1/q5_1 need to fold their block minimum into the dot.
- **Q8_K** (`quantizeRowQ8_KInto`, `quantizeRowsQ8_K`): per-256 with an f32
  scale derived from the signed maximum (`inv_scale = -127/max`) and
  per-16-element partial sums (`bsums`) — the K-quant and ternary kernels
  consume the bsums to fold block minima / the ternary `−Σa` term for free.
  Packed-LHS variants (`quantizeRowsQ8_Kx4Into`,
  `quantizeRowsQ8_Kx4PaddedInto`, `quantizeRowsQ8_Kx2MmlaInto`,
  `packRowsQ8_Kx4`) interleave 4 (or 2 for MMLA) rows for the x4/x2mmla
  kernels.

Scratch policy: only the **Q8_0/Q8_0x4** LHS buffer has a stack fast path —
a fixed 512-block array (`q8_0_lhs_stack_blocks`) covers GEMVs and small
batches with zero allocation, falling back to a context-allocator heap
allocation when `m × blocks_per_row` exceeds it. The Q8_K path (every
K-quant, IQ\*, and TQ\* weight) and the Q8_1 path heap-allocate their block
buffers unconditionally (`quantizeRowsQ8_K`/`quantizeRowsQ8_1` take the
allocator). The native backend's fused K-quant split-SwiGLU and GeGLU paths
([§10.3](10-quantization.md#103-rhs-containers-and-packed-layouts-srcbackendquanttypeszig-srcexecquant_matmulzig)) lease buffers from the context's reusable scratch pool
(`buffers.acquireScratch`) — the q8_0x4 fused split-SwiGLU arm puts a
512-block stack array in front of the same pool lease — and parallelize row-group
quantization across the work pool once `m·k` reaches **one eighth** of
`parallel.vector_elementwise_len_threshold`. The unfused native dispatch tier
(`matmul2DQuantizedRhs*` / `matmulPacked` in `src/backend/native.zig`) splits
its own LHS quantization the same way: the allocation-free range forms
(`quantizeRowsQ8_0RangeInto`, `quantizeRowsQ8_KRangeInto`,
`quantizeRowsQ8_0x4GroupsInto`/`quantizeRowsQ8_0x4PaddedGroupsInto`,
`quantizeRowsQ8_Kx4GroupsInto`, `quantizeRowsQ8_Kx2MmlaGroupsInto`) quantize a
row or row-group range, and the tier fans those ranges over the caller's
`ParallelConfig` pool at the same one-eighth gate, so a decode row never
touches the pool and every split is bitwise the serial walk (rows own
disjoint blocks). The per-row Q8_0 and Q8_K bodies are one `@Vector` form on
every ISA (aarch64 keeps its `fcvtas`/`fcvtns` rounding legs; elsewhere the
portable vector round and the 2^23 magic-number round-half-even), byte-equal
to the scalar `quantizeToI8`/`roundNearestEven` forms that
`src/x86dot_check.zig` pins them against. Failure mode: allocation
failure is the only runtime error; quantization itself is total for finite
input.

`ExecContext` also exposes the Q8_0 activation codec directly —
`quantizeF32RowsToQ8_0Into(x, dst_blocks)` and
`dequantizeQ8_0RowsInto(dst, blocks)` — used to maintain Q8_0 KV caches for
`groupedAttention`'s quantized-KV representation ([§4](04-tensor-operations.md)).

## 10.6 Encoders, `gguf.encodeF32`, and ggml parity (`src/backend/quant.zig`, `src/gguf.zig`)

The kernel tier's encoder dispatch:

```zig
pub fn quantizeRowForDType(comptime tensor_dtype: DType, dst: []dtype_mod.Storage(tensor_dtype), src: []const f32) !void
pub fn dequantizeRowForDType(comptime tensor_dtype: DType, dst: []f32, blocks: []const dtype_mod.Storage(tensor_dtype)) !void
pub fn blockCountForDType(comptime tensor_dtype: DType, len: usize) !usize
```

`quantizeRowForDType` covers `.q2_0`, `.q4_0`, `.q4_1`, `.q5_0`, `.q5_1`, `.q8_0`,
`.q8_1`, `.q4_k`, `.q5_k`, `.q6_k`, `.q8_k`, `.tq2_0`; any other dtype is a
compile error. `src.len` must be a whole number of blocks and `dst.len` must
equal `blockCountForDType(dtype, src.len)` —
`QuantizedFormatError.InvalidQuantizedLength` otherwise. Inputs are assumed
**finite** (same contract as ggml's encoders; debug asserts mirror ggml's
`nearest_int` bound). The K-quant encoders are operation-for-operation ports
of `quantize_row_{q4,q5,q6}_K_ref`: the shared iterative scale-search
helpers `makeQxQuants` (symmetric, Q6_K) and `makeQkx2Quants` (asymmetric
scale+min grid search, Q4_K/Q5_K) plus `nearestInt` (round-to-nearest-even
via the 1.5·2²³ magic constant) and `group_max_eps` reproduce ggml's f32
arithmetic exactly. Per-block entries also exist
(`quantizeBlockQ4_KInto`/`Q5_K`/`Q6_K`, `dequantizeBlockQ4_KInto`/`Q5_K`/
`Q6_K`/`Q8_K`/`Q2_K`/`Q3_K`, `getScaleMinK4`).

The public seam is `fucina.gguf.encodeF32(ggml_type, src, dst)` ([§12](12-model-io-gguf-and-safetensors.md)): it
validates `dst.len == tensorByteLen(...)`, rejects non-finite input on the
block formats with `error.NonFiniteValue` (release builds too — the same
guard llama.cpp applies at its quantize seam), requires `dst` to be aligned
for the block struct, and dispatches to `quantizeRowForDType`. Supported
block targets: q2_0, q4_0, q4_1, q5_0, q5_1, q8_0, q4_k, q5_k, q6_k, tq2_0
(scalar f32/f16/bf16 cast element-wise); everything else returns
`error.EncoderUnavailable`. `gguf.decodeF32` is the exact mirror
(`error.DecoderUnavailable` for formats it does not cover).

Parity evidence, all in-tree and run by `zig build test`:

- `src/backend/quant/encode_golden_test.zig` — embedded goldens generated
  once by a C harness linking ggml's reference encoders over 8 adversarial
  input vectors (ramp, alternating, near-zero, wide-range, all-equal,
  denormals, random, zeros); the Zig encoders for Q4_K/Q5_K/Q6_K/Q4_1/Q5_0/
  Q5_1 match **byte-for-byte**, and the oracle was verified stable across
  three compiler/FP-contraction configurations.
- `src/backend/quant/cold_tests.zig` — embedded ggml-golden dequantize
  fixtures reproduced **bit-for-bit** for every cold decode format
  (Q2_K/Q3_K, all IQ*, TQ*, MXFP4/NVFP4), plus behavioral matmul tests for
  the table-dot paths.
- `src/backend/quant_tests.zig` and the per-format
  `quant/{q4_k,q5_k,q6_k,q8_0,common}_tests.zig` — hot-kernel vs scalar
  reference equivalence; `src/backend/quant/ternary_tests.zig` pins the
  ternary hot kernels **bitwise** against the cold scalar reference.

## 10.7 Ternary: TQ2_0 first-class, TQ1_0 decode-only (`src/backend/quant/ternary.zig`, [TERNARY.md](../TERNARY.md))

TQ2_0 (BitNet b1.58: weights in {−1, 0, +1}, 2.0625 bits/weight — 256-element
blocks, 64 bytes of 2-bit crumbs storing `w+1`, inline f16 scale) is promoted
to a first-class format: encoders, tuned mul-free kernels on both the int8
and f32 activation paths, a facade training op, ternary-native ES ([§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md)), and
GGUF export interop with llama.cpp. The contract dimension `k` must be a
multiple of 256 everywhere.

Encoders:

- `quantizeRowTQ2_0Into` — ggml `quantize_row_tq2_0_ref` parity: per-block
  absmax `d`, round-half-away(x/d). This is the **only** encoder behind the
  generic seams: `quantizeRowForDType(.tq2_0, ...)` dispatches to it, and
  `gguf.encodeF32(.tq2_0, ...)` therefore realizes per-block absmax too.
- `ternaryAbsmeanScale` + `quantizeRowTQ2_0ScaledInto` — the b1.58 recipe:
  per-tensor `d = max(mean|W|, 1e-5)`, `clamp(round(W/d), −1, +1)`; every
  block stores the same `d`, so the output is plain valid TQ2_0. This
  scaled encoder takes an explicit `d` and is *not* reachable through
  `quantizeRowForDType`/`gguf.encodeF32` — it is driven by
  `quantizedMatmulRhsTQ2_0FromF32Absmean` (the `dotTernarySte` forward,
  [§4.9](04-tensor-operations.md#49-explicit-matmul-ternary-ste-and-packed-rhs-gemms-srcagtensorzig)) and the ternary ES paths ([§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md)).

Kernels: the int8 flagship exploits `dot(w, a) = Σ(w+1)·a − Σa` — the Q8_K
activation `bsums` supply `Σa`, so the hot loop is shift/mask plus int8
group dots with **no weight multiplications** (`matmulTQ2_0RhsTile`/
`matmulTQ2_0RhsRange`); all ISA arms accumulate the exact per-block integer
and are cross-ISA bitwise identical to the cold reference. The f32 path
(`dotTQ2_0F32`, `matmulTQ2_0F32RhsTile`/`Range`) is exact in IEEE f32 via
the sign-plane/zero-plane identity `w·x = (x XOR s) AND m` with a fixed
4-lane accumulation order — bitwise reproducible on every target, used for
STE training forwards. RHS constructors:
`quantizedMatmulRhsTQ2_0FromBorrowedBlocks` (borrows, `deinit`
frees nothing), `quantizedMatmulRhsTQ2_0FromF32` (absmax) and
`quantizedMatmulRhsTQ2_0FromF32Absmean` (b1.58). Measured: ~4.2x the cold
path and ~2.1x Q4_K per byte-ratio on M1 Max; ~5.1x cold and ~4.8x Q4_K on
Raptor Lake AVX-VNNI ([TERNARY.md](../TERNARY.md)).

A kernel-tier x4 pack exists below the facade: `packMatmulRhsTQ2_0x4`
rearranges four RHS rows (output columns) into `BlockTQ2_0x4` groups,
column-interleaved in 4-byte granules so one 16-byte load feeds the same
4-element k-group to all four columns — each int8 dot lane accumulates
its own column and the per-block horizontal reduce disappears (the
Q8_0x4/Q4_Kx8 layout family, ternary form).
`matmulTQ2_0X4RhsTile`/`matmulTQ2_0X4RhsRange` consume it against the
same Q8_K activations, bit-identical to the row kernels; the accumulating
twin `matmulTQ2_0X4RhsTileAcc` folds additional planes straight into the
output. `n % 4 == 0` is required, and there is no facade entry (`packRhs`
on a `.tq2_0` tensor stays a compile error) — the LLM layer's PTQTP
wrappers build and consume the packs ([§13.2.1](13-the-model-stack-fucina_models.md#132-weight-loading-srcweightszig)). For scale-tied PTQTP
pairs ([§10.9](10-quantization.md#109-ptqtp-multi-plane-ternary-decomposition-srcptqtpzig-ptqtpmd)), the fold family (`packMatmulRhsTQ2_0Foldedx4`/`Foldedx4Into`,
`packMatmulRhsTQ2_0FoldedRows`, kernels
`matmulTQ2_0FoldedX4RhsTile`/`Range`) combines two ratio-3 planes into
one 4-bit code pack (`BlockTQ2_0Foldedx4`; row-major `BlockTQ2_0Folded`
for the GPU dequant-in-kernel GEMM) — codes `c = 3·t1 + t2` in {0..8}
with the fine plane's f16 scale, the coarse scale derived as 3× in f32 —
so a single dot pass serves both planes.

As inference weights, `.tq2_0` tensors go through the ordinary facade:

```zig
test "TQ2_0 ternary weights are a first-class matmul RHS" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const k = 256;
    var wsrc: [2 * k]f32 = undefined;
    for (wsrc[0..k]) |*v| v.* = 0.5; // encodes exactly: d = 0.5, trit = +1
    for (wsrc[k..]) |*v| v.* = -0.5;
    var blocks: [2]fucina.quant.BlockTQ2_0 = undefined;
    try fucina.gguf.encodeF32(.tq2_0, &wsrc, std.mem.sliceAsBytes(&blocks));

    const W = fucina.Tensor(.{ .dtype = .tq2_0, .tags = .{ .out, .in } });
    var w = try W.fromBlocks(&ctx, .{ 2, k }, &blocks);
    defer w.deinit();

    const x_values = [_]f32{1} ** k;
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 1, k }, &x_values);
    defer x.deinit();

    var y = try x.dot(&ctx, &w, .in); // mul-free int8 ternary kernel
    defer y.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 128), (try y.dataConst())[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -128), (try y.dataConst())[1], 1e-3);
}
```

### `dotTernarySte` — the straight-through-estimator training op

```zig
pub fn dotTernarySte(self: *const Self, ctx: *ExecContext, weight: anytype,
                     comptime contract_tag: Tag) !Tensor(...)
```

Trainable ternary linear on the f32 facade tensor (`src/ag/tensor.zig`).
Every forward encodes the **f32 latent weight** (tags `{ .out, .in }`,
per-tensor absmean scale, round-clip to {−1, 0, +1}) to TQ2_0 and contracts
`self` (`[..., in]`) against it with the mul-free f32 kernel — the same
`@Vector` kernel on both backend kinds, so scalar and native builds share
bitwise-identical numerics. Backward: `dx` flows through the **quantized**
weight (dequantize-then-matmul); `dW` is the straight-through estimate — the
plain matmul VJP against the latent weight, identity through the quantizer,
no clipping or masking (exactly BitNet's `w + (Q(w) − w).detach()`). The
encoded blocks live in the backward node and are freed with it; the op works
under exec scopes.

Comptime requirements: f32 latent weight, storage order `[free, contract]`
on the weight and `[..., contract]` on the LHS, one weight free axis, no
shared batch tags. Runtime: `error.TernaryContractDimNotBlockAligned` when
`k` is 0 or not a multiple of 256. The latent weight is re-encoded every
forward (inherent to STE).

```zig
test "dotTernarySte trains a b1.58 ternary linear" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const k = 256; // contract dim must be a multiple of 256 (TQ2_0 blocks)
    var xv: [k]f32 = undefined;
    for (&xv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5)) * 0.1 - 0.2;
    var wv: [2 * k]f32 = undefined;
    for (&wv, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 9)) * 0.1 - 0.4;

    var x = try fucina.Tensor(.{ .batch, .in }).variableFromSlice(&ctx, .{ 1, k }, &xv);
    defer x.deinit();
    var w = try fucina.Tensor(.{ .out, .in }).variableFromSlice(&ctx, .{ 2, k }, &wv);
    defer w.deinit();

    var y = try x.dotTernarySte(&ctx, &w, .in); // encode-then-mul-free forward
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();
    try loss.backward(&ctx);

    // STE identity: dW is the plain matmul VJP; with gy = 1, each row of dW = x.
    var gw = (try w.grad(&ctx)).?;
    defer gw.deinit();
    const gw_data = try gw.dataConst();
    for (0..k) |i| try std.testing.expectApproxEqAbs(xv[i], gw_data[i], 1e-6);
}
```

The gradient-free alternative is **ternary-native evolution strategies**
([§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md)): ES genomes that *are* packed `[]BlockTQ2_0` — no latent floats, so the
trained state is byte-for-byte the served state and every member evaluation
runs the real int8 inference kernels (`examples/es_ternary_spirals/main.zig` is
the end-to-end demo). `.tq1_0` (1.6875 bits/weight, base-3⁵ packing of five
trits per byte) remains decode/cold-matmul only.

### Q2_0 — the Bonsai g128 ternary container

`.q2_0` (ggml type 42, PrismML/Bonsai `Q2_0_g128`) is TQ2_0's sibling with a
different envelope: 128-element blocks (34 bytes: f16 absmax scale + 32 bytes
of 2-bit codes packed four-per-byte LSB-first), codes `q` in {0,1,2,3}
decoding to `(q-1)·d` — the reference encoder only ever emits {0,1,2}
(round against the block absmax), so files are pure ternary; code 3 = +2d is
wire-contract-only. It is first-class like TQ2_0: a parity encoder
(`quantizeRowQ2_0Into`, reachable through `gguf.encodeF32`), decode, and hot
mul-free kernels (`matmulQ2_0RhsTile/Range`, `src/backend/quant/ternary.zig`)
that pair with **Q8_0 row activations** (so `k` needs only be a multiple of
128): unsigned-code dots through sdot/vpdpbusd/maddubs with one per-32-group
bsum subtraction (`Σ(q-1)a = Σq·a − Σa`), bsums and activation scales cached
once per LHS row and shared across all output columns, two LHS rows sharing
every weight unpack. All arms accumulate the exact per-group i32, and the
float tail is one vector FMA per 128-block over a **fixed 4-lane sub-block
accumulator** folded pairwise once per output element — exactly the scalar
reference's contract (`dotQ2_0RowQ8_0` / `matmulQ2_0RhsRefRange`, cold.zig;
the scalar backend's path), so hot and reference are bitwise identical. At
prefill widths (`m >= 192`) the native backend instead dequantizes k-slice
weight panels to f32 (`dequantizeRowQ2_0FastInto`, the kernels' unpack
machinery) and rides the BLAS GEMM with `beta=1` accumulation across
slices — panels cut the contract dimension so every GEMM stays full-width
with a contiguous C. The BLAS arm consumes exact f32 activations, so its
numerics differ from the int path the same way the dense-f32 BLAS GEMMs
differ from the scalar backend. This is the weight format of
Ternary-Bonsai-27B ([§14.3](14-model-families-and-example-applications.md#143-qwen35--gated-deltanet-hybrid-srcmodelsqwen35modelzig)).

## 10.8 Cold decode rules: IQ*, FP4, and friends (`src/backend/quant/cold.zig`, `src/backend/quant_tables.zig`)

`quant_tables.zig` holds the lookup tables generated from ggml's
`ggml-common.h` (`iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`,
`iq3s_grid`, `iq1s_grid`, `ksigns_iq2xs`/`kmask_iq2xs`, `kvalues_iq4nl`,
`kvalues_mxfp4`). Decode semantics, one line each:

- **Q1_0** — pure sign bits: bit set → `+d`, clear → `−d` (f16 per-block
  scale, 128 weights/block).
- **IQ2_XXS / IQ2_XS / IQ2_S** — 2.06–2.5 bpw: 8-element groups are indices
  into a shared 256/512/1024-entry codebook grid of magnitude patterns
  {8, 25, 43}, with separate sign words (`ksigns_iq2xs`) and 4-bit group
  scales times the f16 block scale.
- **IQ3_XXS / IQ3_S** — same grid-codebook construction at ~3 bpw with
  4-element grids (`iq3xxs_grid`/`iq3s_grid`) and explicit sign bits.
- **IQ1_S / IQ1_M** — 1.56/1.75 bpw: 11-bit indices into the 2048-entry
  8-element grid `iq1s_grid`, per-32 3-bit scales, and a whole-group
  ±0.125 delta added to every element (IQ1_M drops the f16 block scale and
  packs all scales into nibble fields).
- **IQ4_NL / IQ4_XS** — 4-bit **nonlinear** codebook: nibbles index the
  16-entry `kvalues_iq4nl` table (an asymmetric nonlinear ladder from −127
  to 113) times an f16 block scale; IQ4_XS adds per-32 sub-scales inside
  256-blocks.
- **MXFP4** — OCP MX: one shared **E8M0** exponent byte per 32 elements
  (decoded as a pure power of two, halved to compensate the
  integer-doubled table) times the 16-entry FP4 **E2M1** codebook
  `kvalues_mxfp4` = {0, ±1, ±2, ±3, ±4, ±6, ±8, ±12}.
- **NVFP4** — NVIDIA FP4: four 16-element sub-blocks per 64-element block,
  each with a **UE4M3** (unsigned e4m3) scale byte, same E2M1 codebook.
- **TQ1_0** — five trits packed per byte in base-3 (with a small `qh`
  tail), f16 scale.

All of these dequantize (`to(.f32)`, `getRows`) and matmul through the
generic cold dot path with Q8_0- or Q8_K-quantized activations per the
[§10.1](10-quantization.md#101-format-inventory-srcbackendquanttypeszig-srcdtypezig) table; none has an encoder — they enter Fucina only as loaded GGUF
weights ([§12](12-model-io-gguf-and-safetensors.md)) and their decode output is pinned bit-for-bit to ggml goldens
(`cold_tests.zig`).

## 10.9 PTQTP: multi-plane ternary decomposition (`src/ptqtp.zig`, [PTQTP.md](../PTQTP.md))

`fucina.ptqtp` decomposes a dense f32 `[n][k]` weight matrix into
K ∈ {1, 2, 3} ternary planes, `W ≈ Σₖ diag(αₖ)Tₖ` with `Tₖ ∈ {−1, 0, +1}`
and one scale per plane per 256-column group — data-free post-training
quantization (arXiv:2509.16989; no calibration inputs, no gradients). Each
group solves independently by alternating a closed-form K×K ridge
regression for the scales with an exhaustive 3ᴷ-way per-element trit
search, in a pinned candidate order that makes the whole solve bitwise
reproducible for any thread count. The group size is the TQ2_0 block
width, so **each plane is a standalone byte-valid `.tq2_0` tensor** whose
per-block f16 `d` is the group scale: decorated inference is K ternary
matmuls ([§10.7](10-quantization.md#107-ternary-tq2_0-first-class-tq1_0-decode-only-srcbackendquantternaryzig-ternarymd)) plus adds — through the plain row kernels, or the x4
column-interleaved kernels and their accumulating twin once the serving
layer has built packs; a tie-fitted K = 2 pair additionally folds into
one 4-bit pack that a single dot pass serves ([§10.7](10-quantization.md#107-ternary-tq2_0-first-class-tq1_0-decode-only-srcbackendquantternaryzig-ternarymd)). `k % 256 == 0` is
required.

```zig
pub const Options = struct {
    planes: u8 = 2,            // K; 3 adds a residual plane (+2.06 bpw)
    group_size: usize = 256,   // quantizeMatrix requires 256 (the packable size)
    max_iterations: usize = 50, epsilon: f32 = 1e-4,
    lambda0: f32 = 1e-8, lambda_max: f32 = 1.0, kappa_max: f32 = 1e6,
    tie_scales: bool = false,  // lock plane scales to the exact ratio 3 (requires planes >= 2)
};
pub fn quantizeMatrix(ctx: *ExecContext, weights: []const f32, n: usize, k: usize,
                      options: Options) !PlanePair
pub fn solveGroup(w: []const f32, t1: []i8, t2: []i8, t3: []i8,
                  options: Options) GroupResult   // pure, allocation-free
pub fn reconstructReference(allocator, weights, n, k, options, dst: []f32) !MatrixStats
```

`PlanePair` owns up to three `[]BlockTQ2_0` planes (`plane2`/`plane3`
empty below the built count) plus `stats: MatrixStats` (rel Frobenius
error of the served reconstruction — fp16-rounded scales — per-plane zero
fractions, iteration/convergence counts). `rhs(plane)` returns a borrowed
backend matmul view; `reconstructInto(dst)` writes the dequantized plane
sum; `deinit(allocator)` frees the planes. `reconstructReference` solves
at arbitrary group sizes with f32 scales (unpacked) for fidelity studies.
Non-finite weights are excluded from the regression and forced to zero
trits in every plane. K = 1 is a least-squares upgrade over the absmean
b1.58 encoder ([§10.7](10-quantization.md#107-ternary-tq2_0-first-class-tq1_0-decode-only-srcbackendquantternaryzig-ternarymd)); planes are separable, so serving fewer planes than
were solved is valid.

`tie_scales` locks the plane scales to the exact ratio 3 (`alpha = [3s, s]`
at K = 2, `[9s, 3s, s]` at K = 3), turning the K planes into one uniform
symmetric 3ᴷ-level quantizer with step `s`; the tied solve replaces the
ridge iteration with a per-group step-size sweep against the group
absmax, and it makes the plane-folding identity exact — one combined dot
pass can serve all K planes ([§10.7](10-quantization.md#107-ternary-tq2_0-first-class-tq1_0-decode-only-srcbackendquantternaryzig-ternarymd)). Meaningless at `planes = 1`
(`Error.InvalidOptions`).

```zig
test "PTQTP: planes reconstruct and multiply as plain TQ2_0 tensors" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    const n = 2;
    const k = 256;
    var w: [n * k]f32 = undefined; // any f32 [n][k] with k % 256 == 0
    for (&w, 0..) |*v, i| v.* = 0.02 * @sin(@as(f32, @floatFromInt(i)));

    var pair = try fucina.ptqtp.quantizeMatrix(&ctx, &w, n, k, .{}); // K = 2
    defer pair.deinit(ctx.allocator());
    try std.testing.expect(pair.stats.rel_frob_err < 0.25); // 9-level regime

    // Each plane is a byte-valid TQ2_0 tensor; the decorated product is
    // the sum of per-plane products through the stock ternary kernels.
    const W = fucina.Tensor(.{ .dtype = .tq2_0, .tags = .{ .out, .in } });
    var p1 = try W.fromBlocks(&ctx, .{ n, k }, pair.plane1);
    defer p1.deinit();
    var p2 = try W.fromBlocks(&ctx, .{ n, k }, pair.plane2);
    defer p2.deinit();

    const ones = [_]f32{1} ** k;
    var x = try fucina.Tensor(.{ .batch, .in }).fromSlice(&ctx, .{ 1, k }, &ones);
    defer x.deinit();
    var y1 = try x.dot(&ctx, &p1, .in);
    defer y1.deinit();
    var y2 = try x.dot(&ctx, &p2, .in);
    defer y2.deinit();
    var y = try y1.add(&ctx, &y2);
    defer y.deinit();

    var rec: [n * k]f32 = undefined; // the exact Ŵ the stats measured
    try pair.reconstructInto(&rec);
    for (try y.dataConst(), 0..) |yi, r| {
        var want: f32 = 0;
        for (rec[r * k ..][0..k]) |v| want += v;
        try std.testing.expectApproxEqAbs(want, yi, 1e-2);
    }
}
```

At the LLM layer, `LinearWeight.toPtqtp` decorates a loaded GGUF linear in
place from any source dtype and `models.qwen3.ptqtp.decorate`
walks a whole model ([§13.2.1](13-the-model-stack-fucina_models.md#132-weight-loading-srcweightszig)); `zig build ptqtp-spirals` and
`zig build ptqtp-qwen3` are the acceptance/measurement examples. Decorated
models persist to GGUF as per-plane standalone TQ2_0 tensors and load back
bitwise through plane pair-detection in the qwen3 loaders
(`fucina.ptqtp_gguf`, [§13.2.1](13-the-model-stack-fucina_models.md#132-weight-loading-srcweightszig)), so the solve runs once per model; a
scale-tied fit persists as the file key `fucina.ptqtp.tie_scales`
(present only when every decorated linear was tie-fitted), which lets the
loaders rebuild the folded one-pass serving form.
`zig build export-gguf -- --ptqtp[=K] --ptqtp-tie` runs the same solve
shard-streaming, tensor-at-a-time, for models bigger than RAM ([§12.4](12-model-io-gguf-and-safetensors.md#124-the-export-gguf-tool-toolsexport_ggufzig)).
Measured accuracy/speed tables and
configuration guidance (plane counts, source precision, lm_head economics
per ISA): [PTQTP.md](../PTQTP.md).

## 10.10 Fake-quantization round trips (`src/exec/fakequant.zig`)

`fucina.fakequant` passes f32 values through a low-precision grid and back —
quantization-aware *inference* numerics for models whose reference stores
activations or cache rows through such grids, so the round trip is part of
the graph, not an optimization (DeepSeek V4's FP8 KV rows and FP4/Hadamard
indexer QAT, [§13](13-the-model-stack-fucina_models.md)). All kernels run in place over host slices, SIMD and
scalar paths evaluate the same per-element expression (bit-identical for
any length), and the grid rounding is round-to-nearest ties-to-even
implemented arithmetically on the f32 bit pattern — pinned bit-for-bit
against a grid-search oracle in `fakequant_tests.zig`.

- `roundE4m3(x)` / `roundE2m1(x)` — scalar RNE onto the FP8 E4M3 grid
  (saturating at ±448) or the FP4 E2M1 grid (±6). Out-of-range magnitudes
  clamp; no NaN/inf encodings are produced.
- `groupRoundTripE4m3InPlace(x, group_size, amax_floor)` /
  `groupRoundTripE2m1InPlace(...)` — the microscaling recipe: per group,
  `amax = max |x|` floored at `amax_floor` (keeps the scale finite on
  all-zero groups), power-of-two scale `2^ceil(log2(amax / grid_max))`,
  clamp, grid round trip, rescale. `x.len % group_size == 0` is required.
  DeepSeek V4 uses (64, 1e-4) for FP8 KV rows and (32, 6·2⁻¹²⁶) for FP4
  activations.
- `hadamardInPlace(x)` — fast Walsh-Hadamard transform scaled by
  1/sqrt(len) (power-of-two length): the orthonormal rotation used by
  rotation-based quantization schemes (QuaRot/SpinQuant-style QAT).
- `roundF16InPlace(x)` — f32 → f16 → f32 (what storing a row into an f16
  cache does).

The group recipes are **not projections**: a second pass may pick a smaller
power-of-two scale (the amax shrinks through rounding) and re-clamp near
the grid max, so don't assume idempotence.

```zig
test "fake-quant round trips snap values onto their grids" {
    const fq = fucina.fakequant;
    try std.testing.expectEqual(@as(f32, 0.5), fq.roundE2m1(0.6));
    try std.testing.expectEqual(@as(f32, 448.0), fq.roundE4m3(1.0e9)); // saturates
    try std.testing.expectEqual(@as(f32, 4.0), fq.roundE2m1(5.0)); // tie -> even mantissa

    var row = [8]f32{ 0.1, -2.3, 7.5, 0, 1.25, -0.6, 3.9, 448.0 };
    fq.groupRoundTripE4m3InPlace(&row, 8, 1.0e-4);
    for (row) |v| try std.testing.expectEqual(fq.roundE4m3(v), v); // on-grid after the round trip (scale 2^0 from amax 448)

    var had = [4]f32{ 1, 2, 3, 4 };
    fq.hadamardInPlace(&had); // { (1+2+3+4), (1-2+3-4), (1+2-3-4), (1-2-3+4) } / 2
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), had[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), had[1], 1e-6);
}
```
