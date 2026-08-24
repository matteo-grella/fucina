# 9. Backends: CPU SIMD, BLAS, threading, and GPU offload

A backend is the layer that owns numeric kernels and nothing else. `ExecContext`
([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)) validates shapes, allocates outputs, and dispatches; the backend fills
caller-supplied buffers. There are exactly two backends — a scalar reference
and the production `native` backend — selected at build time, plus two optional
accelerator tiers layered *inside* the native backend: platform BLAS for large
f32 GEMM and a GPU GEMM provider (Metal or CUDA). Nothing about backend choice
is visible in tensor types or the public API surface beyond a handful of
comptime constants.

## 9.1 Build-time selection and the facade constants (`src/backend.zig`, `build.zig`)

```zig
pub const Kind = enum { scalar, native };

pub const active_kind: Kind = switch (build_options.backend_kind) {
    .scalar, .cpu => .scalar,
    .native => .native,
};

const active = switch (build_options.backend_kind) {
    .scalar, .cpu => scalar_impl,   // src/backend/cpu.zig
    .native => native_impl,         // src/backend/native.zig
};

pub const ParallelConfig = active.ParallelConfig;
pub const kernels = active.kernels;   // the selected kernel set (§9.2)
```

Selection is a comptime `switch` over `build_options.backend_kind`
(`-Dbackend=native|scalar`).
The inactive implementation is never dispatched to — `backend.kernels` is
the active module's `kernels` namespace and the other module's code paths
are dead — and the switches are exhaustive, so adding a backend variant
forces edits at every dispatch site rather than silently falling through. The same pattern selects
the GPU provider (`src/backend/gpu.zig`): a comptime switch over
`build_options.gpu_kind` resolves `gpu_impl` to `metal.zig` or `cuda.zig`, and
the unselected provider is parsed but never semantically analyzed, so it costs
nothing and needs none of its target's libraries (`cuda.zig` is fully inert on
macOS builds and vice versa). A `comptime` guard in `gpu.zig` verifies
`build_options.use_gpu == (gpu_kind != .none)`.

Build options that shape the backend (see [§2](02-toolchain-build-and-project-wiring.md) for the full option list):

| Option | Values | Default | Effect |
|---|---|---|---|
| `-Dbackend` | `native`, `scalar` | `native` | backend implementation |
| `-Dblas` | `none`, `accelerate`, `openblas`, `mkl`, `blis`, `nvpl`, `blas` | `accelerate` on macOS, else `none` | CBLAS provider for large f32 GEMM; `none` selects the pure-Zig blocked packed GEMM |
| `-Daccelerate` | bool | — | compatibility alias: `false` ≡ `-Dblas=none`, `true` selects Accelerate on macOS |
| `-Dblas-threads` | `u32` | 0 | pin the provider's thread count once at first GEMM; 0 keeps the provider default |
| `-Dmax-threads` | 1–64 | 8 | comptime worker-team ceiling *and* runtime default team size (`parallel.vector_max_threads`) |
| `-Dgpu` | `none`, `metal`, `cuda` | `none` | GPU GEMM offload provider; `metal` requires macOS, `cuda` targets Linux (build panics otherwise) |

The facade (`src/fucina.zig`) re-exports the backend's build facts as comptime
constants:

```zig
pub const BackendKind = backend.Kind;                   // enum { scalar, native }
pub const active_backend_kind = backend.active_kind;
pub const native_blas_kind = backend.native_blas_kind;  // the -Dblas enum value
pub const native_uses_blas = backend.native_uses_blas;  // blas_kind != .none
pub const native_uses_accelerate = backend.native_uses_accelerate;
pub const native_blas_threads = backend.native_blas_threads;
pub const supports_q4_k_mmla = backend.supports_q4_k_mmla;
```

`supports_q4_k_mmla` is `true` on aarch64 targets whose feature set includes
`i8mm` (`src/backend/quant/common.zig` `has_aarch64_i8mm`); it decides which
packed Q4_K RHS layout `packRhs` produces ([§9.7](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#97-quantized-matmul-dispatch-packed-rhs-and-the-int8-dot-arms), [§10](10-quantization.md)). All of these are
comptime values — branches on them fold away:

```zig
test "backend build facts" {
    // All comptime-known — fixed by -Dbackend / -Dblas at build time.
    switch (fucina.active_backend_kind) {
        .native => {}, // default: @Vector kernels + optional BLAS/GPU
        .scalar => {}, // reference backend (-Dbackend=scalar)
    }
    if (fucina.native_uses_blas) {
        try std.testing.expect(fucina.native_blas_kind != .none);
    } else {
        try std.testing.expect(fucina.native_blas_kind == .none);
    }
    if (fucina.native_uses_accelerate)
        try std.testing.expect(fucina.native_uses_blas);
    // -Dblas-threads pin (0 = provider default) and the aarch64+i8mm
    // Q4_K smmla capability, both comptime constants.
    const blas_threads: u32 = fucina.native_blas_threads;
    const q4k_mmla: bool = fucina.quant.supports_q4_k_mmla;
    _ = blas_threads;
    _ = q4k_mmla;
}
```

## 9.2 The kernel interface and the kernel contract (`src/backend/interface.zig`, `src/backend.zig`)

```zig
// src/backend/interface.zig
pub const names = [_][]const u8{ "addInto", "addContiguousIntoUnchecked", ... }; // 99 kernels
pub const generic_names = [_][]const u8{ ... };    // 12: take a comptime dtype or op
pub const pool_free_names = [_][]const u8{ ... };  // 18: take no `pc`
pub fn conform(comptime Impl: type) void;          // the single comptime check

// src/backend/cpu.zig and src/backend/native.zig
pub const kernels = struct { ... };                // exactly the names above

// src/backend.zig
pub const kernels = active.kernels;
comptime {
    interface.conform(scalar_impl.kernels);
    interface.conform(native_impl.kernels);
}
```

The two implementations meet at one interface, declared once by name in
`interface.zig`. It is a comptime-checked namespace rather than a struct of
function pointers because many kernels are generic over a `comptime` dtype
or op. `conform` verifies, for every name: the implementation declares it;
a non-generic entry has the native entry's parameter types and return
payload (fallible kernels return `!T` with an inferred error set, and two
inferred sets are distinct types by construction, so the payload is what is
compared); a generic entry has the same parameter count; the `pc` rule below
holds; and the implementation declares nothing beyond the set.

The signature rule: a kernel that needs the worker pool takes
`pc: ParallelConfig` as its FIRST parameter; a kernel that does not use the
pool does not take it (`pool_free_names`). Both implementations follow the
same rule for every name, so an exec caller is backend-agnostic; the scalar
reference accepts `pc` wherever the native kernel threads on it and ignores
it. `ParallelConfig` is `struct { pool: ?*thread.Pool = null }`; `.{}` runs
the kernel serially.

The worker team lives on `ExecContext` (`parallel_pool`, an atomic pointer
published by `tryWorkPool`); every call site reads it through `ctx.pc()`, a
per-call snapshot with acquire ordering ([§6.6](06-the-execution-runtime-execcontext-and-the-memory-model.md#66-the-worker-team-srcthreadzig-srcparallelzig)):

```zig
const kernels = backend_mod.kernels;           // file-level, in each exec/ module
kernels.sumInto(ctx.pc(), &out, &x);           // pool-taking
kernels.matmulInto(&out, &a, &b);              // pool-free
```

Kernel naming encodes the checking tier — with one caveat:

- `...Into(out, ...) !void` — validates shapes itself (`rankView`, dim
  checks) and returns `TensorError.ShapeMismatch` on disagreement. This
  holds for the elementwise/reduction/dot/matmul families (`addInto`,
  `sumInto`, `dotInto`, `matmulInto`, …). The conv/pool/norm families
  (`conv2dInto`, `im2colInto`, `pool2dInto`, `upsample2xNearestInto`,
  `conv1dInto`, `col2im1dInto`, `snakeInto`, `groupNormInto`,
  `causalDepthwiseConv1dInto`, …) are `...Into`-named but plain `void` and
  **unchecked** — the exec layer validates geometry before calling them.
- `...IntoUnchecked` and the slice kernels (`addScaledSlice`,
  `unaryRowSlice`, `preluChannelsInto`, ...) — `void`; the caller (exec) has
  already validated shape and contiguity. Passing wrong geometry is illegal
  (out-of-bounds slice panics in safe builds, UB in ReleaseFast).
- `...Typed` variants take a comptime `DType` and typed tensors
  (`TensorOf(dtype)`), with output dtype derived by
  `dtype_mod.outputDType(...)`.

The full kernel inventory, grouped (every name is an entry of
`interface.names`; entries marked with an asterisk take no `pc`):

| Family | Kernels |
|---|---|
| elementwise | `addInto`*, `addContiguousIntoUnchecked`, `subInto`*, `subContiguousIntoUnchecked`, `mulInto`*, `mulContiguousIntoUnchecked`, `divContiguousIntoUnchecked`, `maximumContiguousIntoUnchecked`, `minimumContiguousIntoUnchecked`, `elementwiseContiguousIntoTyped`, `scaleInto`, `unaryContiguousIntoUnchecked`, `leakyReluContiguousIntoUnchecked`, `clampContiguousIntoUnchecked`, `gatedContiguousIntoUnchecked` |
| row/slice helpers | `addScaledSlice`*, `addRowVectorSlice`*, `addRowVectorUnarySlice`*, `unaryRowSlice`*, `mulRowSlice`*, `preluChannelsInto`, `preluChannelsBackwardInputInto`, `preluChannelsBackwardAlphaInto`, `channelAffineInto` |
| reductions | `sumInto`, `sumSlice`*, `prodInto`, `prodSlice`*, `sumSliceTyped`, `dotInto`, `dotIntoTyped` |
| 1-D conv | `causalDepthwiseConv1dInto` (+`BackwardInputInto`, `BackwardKernelInto`), `causalConv1dInto` (+`BackwardInputInto`, `BackwardWeightInto`), `groupedCausalConv1dInto` (+`BackwardInputInto`, `BackwardWeightInto`), `conv1dInto` (+`BackwardInputInto`, `BackwardWeightInto`), `col2im1dInto`, `col2im1dBackwardInto` |
| 2-D conv / image | `conv2dInto`, `conv2dBackwardInputInto`, `conv2dBackwardWeightInto`, `im2colInto`, `col2imInto`, `pool2dInto`, `avgPool2dBackwardInto`, `maxPool2dBackwardInto`, `upsample2xNearestInto` |
| Winograd transforms | `winogradF2WeightTransformInto`, `winogradF2InputTransformInto`, `winogradF2OutputTransformInto`, `winogradF4WeightTransformInto`, `winogradF4InputTransformInto`, `winogradF4OutputTransformInto` |
| norm / activation kernels | `groupNormInto`, `groupNormBackwardInto`, `snakeInto`, `snakeBackwardInputInto`, `snakeBackwardParamsInto` |
| dense GEMM | `matmulInto`*, `matmul2DIntoUnchecked`, `matmul2DAccIntoUnchecked`, `matmul2DIntoUncheckedTyped`, `matmulTransAInto`*, `matmulTransA2DIntoUnchecked`, `matmulTransBInto`*, `matmulTransB2DIntoUnchecked`, `matmulTransB2DIntoUncheckedF16Operands`, `matmulTransB2DIntoUncheckedBf16Rhs` |
| batched GEMM | `matmulBatched2DIntoUnchecked`, `matmulBatchedTransA2DIntoUnchecked`, `matmulBatchedTransB2DIntoUnchecked` |
| packed dense RHS | `packDenseMatmulRhsTyped`*, `matmul2DIntoUncheckedPackedDenseRhs`, `packMatmulRhsTyped`*, `matmul2DIntoUncheckedPackedRhsTyped` |
| quantized RHS | `quantizeMatmulRhsBlockwiseI8`*, `quantizeMatmulRhsQ4_0`*, `quantizeMatmulRhsQ8_0`*, `matmul2DQuantizedRhs`, `matmulQuantizedRhs` (comptime dtype, the plain K-quant containers), `matmulPacked` (comptime container dispatch over the packed layouts), `matmulPackedSlice` (pre-quantized LHS slices), `matmul2DPackedQ8_0x4LhsRhs`, `matmul2DPackedPaddedQ8_0x4LhsRhs` |

Geometry structs re-exported through `backend.zig` (and used in the
signatures above): `Conv2dDims` (channel-last `[H,W,Cin] → [OH,OW,Cout]`;
fields `h, w, cin, oh, ow, cout, kh, kw, stride_h, stride_w, pad_h, pad_w,
groups`), `Conv1dDims` (`seq, out_len, in_channels, out_channels, taps,
stride, pad, dilation, groups`), `Pool2dDims` (`h, w, c, oh, ow, kh, kw,
stride_h, stride_w, pad_h, pad_w`), `PoolKind = enum { avg, max, sum }`, and
`WinogradF2Dims` (shared by the F2 and F4 transforms).

**The allocation contract**, precisely scoped:

- Output buffers are always supplied by `ExecContext`; no backend allocates
  tensor outputs. The vector/quant compute leaves (`src/backend/vector/*`,
  the dot kernels in `src/backend/quant/*`) are allocation-free.
- The quantized-RHS dispatch tier (`matmul2DQuantizedRhs*` in
  `native.zig`/`cpu.zig`) deliberately takes an allocator for per-call LHS
  quantization scratch (f32 activation rows → `Q8_0`/`Q8_1`/`Q8_K` blocks);
  the Q8_0 arms have a 512-block stack fast path
  (`q8_0_lhs_stack_blocks = 512`) so decode-sized calls allocate nothing.
  RHS pack preparation (the x4/x8 lane packs) allocates at load time, not
  per matmul. The exec-tier packed-LHS scratch above this seam is pooled
  ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)); pooling the backend-tier scratch below the seam is an open,
  bench-gated task.
- Direct native vector kernels accept a `ParallelConfig` so the execution
  context controls thread-pool ownership — a kernel never creates threads
  and never assumes a pool exists (`.pool = null` runs serially).

`src/backend/ops.zig` defines the shared op vocabulary both backends compile
against: `ElementwiseOp` (`add, sub, mul, div, max, min`), `UnaryOp` (`relu,
exp, sqrt, rsqrt, sigmoid, silu, log, log1p, softplus, neg, abs, sin, cos,
tanh, fast_tanh, gelu, quick_gelu, softcap_30, softcap_15, gelu_quant, elu,
gelu_erf, floor, ceil, round, sign, reciprocal`), `GatedOp` (`glu, swiglu,
geglu, swiglu_clamp10, situ` — `situ` is Kimi K3's SiTU: the gate
activation is `4·tanh(g/4)·sigmoid(g)`, a soft-bounded SiLU, and the up
input is soft-clamped to `25·tanh(u/25)` before the multiply), and
`CompareOp` (`eq, ne, lt, le, gt, ge` — exec-level only, no
backend kernel), plus the scalar reference semantics (`unaryScalar`,
`gatedActivationScalar`, `compareScalar` with IEEE-754 NaN rules, and `erff`,
a faithful musl translation so `gelu_erf` matches `ggml_vec_gelu_erf_f32`).
`gelu_quant` reproduces ggml's f16-LUT GELU bit-for-bit (input and output
rounded through f16, hard clamps at ±10) for llama.cpp numeric parity; `gelu`
is the exact tanh-approximation form.

## 9.3 The scalar backend and the parity contract (`src/backend/cpu.zig`, `src/backend/parity_test.zig`)

The scalar backend is the numeric reference: plain serial loops, no SIMD, no
BLAS, no GPU. Its `kernels` namespace has the same signatures as the native
one (`interface.conform` checks it) and ignores `pc` (every kernel is
serial). Where a
kernel is *routing shared by both backends* rather than divergent numerics —
`im2col`, the Winograd weight/input/output transforms, the conv2d backward
loops, the pool2d backward scatters — the scalar backend reuses the shared
correctness-first implementation serially, so the two backends compute
identical values on those paths by construction; everything with a real SIMD
counterpart (elementwise, reductions, GEMM, pool2d forward, prelu/affine) is
an independent scalar implementation.

`src/backend/parity_test.zig` imports **both** `cpu.zig` and `native.zig`
directly (independent of `-Dbackend`), so `zig build test` always runs the
cross-backend parity suite. What it guarantees:

- `addInto`, `subInto`, `mulInto`, `scaleInto` agree within `1e-6` absolute
  over lengths `{1, 3, 7, 8, 15, 16, 17, 31, 64, 128, 257, 1024}` (edge cases
  around every vector width) plus a 300 000-element case for
  `addInto`/`mulInto`/`scaleInto` that crosses the parallel-split thresholds.
- `sumInto`, `dotInto` agree within `1e-6·n` (the SIMD pairwise/parallel
  reduction reassociates; tolerance scales with the accumulation count).
- `matmulInto`, `matmulTransAInto`, `matmulTransBInto` agree within `1e-5·k`
  over shapes up to `64×64×64` plus `48×192×128`.
- The batched GEMM triple (`matmulBatched2DIntoUnchecked`,
  `...TransA...`, `...TransB...`) agrees over batch counts `{1, 2, 5, 8}`,
  including broadcast RHS (`stride_b = 0`) and shared LHS (`stride_a = 0`).
- `pool2dInto` (max/avg/sum, odd channel counts to exercise SIMD
  remainders), `upsample2xNearestInto`, `preluChannels*`, and
  `channelAffineInto` (with and without shift) agree within `1e-6`.

The suite pins the semantics the native backend must preserve while its
kernels are rewritten for speed; anything not covered by shared routing or a
parity test is covered by the op-level tests in `backend_tests.zig` and the
exec-layer suites.

## 9.4 Native backend: portable `@Vector` kernels (`src/backend/vector/`)

The native backend's non-GEMM work is pure Zig `@Vector` code, portable
across NEON, AVX2/AVX-512, and WASM SIMD. The vector width is chosen at
comptime by `std.simd.suggestVectorLength(f32) orelse 4` — 4 lanes on NEON,
8 on AVX2, 16 on AVX-512 — with separate widths for f16 and f64
(`vector_len_f16`, `vector_len_f64`). `vector.zig` exposes the modules by
name (`vector.gemm.matmul2DIntoUnchecked`) plus `ParallelConfig`, `Vf32`
and `vector_len`; every pool-taking kernel takes `pc: ParallelConfig`
first. Module map:

| Module | Contents |
|---|---|
| `vector/common.zig` | shared leaf: `ParallelConfig`, `V*` width aliases, thread-count gates, contiguous-data accessors |
| `vector/primitives.zig` | `@Vector` leaves: `vecAdd/Sub/Mul/Scale/AddScaled`, `vecUnary/AddUnary/LeakyRelu/Clamp/Gated`, `vecUnaryVjp` (unary-VJP derivative bodies; `unaryVjpVectorizes` gates which ops route here — the exp-family backward loops), `vecSum/Dot`, f16/bf16/f64 typed twins, `vexpf`, bf16 bit converters |
| `vector/elementwise.zig` | elementwise/reduction entry points, parallel dispatch, snake/groupNorm/prelu/channelAffine kernels |
| `vector/gemm.zig` | dense f32/f16/f64/bf16 GEMM (NN/TN/NT), register-tiled row kernels, `gemmNNRange/gemmTNRange/gemmNTRange` |
| `vector/gemm_blocked.zig` | BLIS-style cache-blocked packed f32 GEMM ([§9.5](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#95-gemm-dispatch-precedence-blas-and-the-blocked-packed-kernel-srcbackendnativezig-vectorgemmzig-vectorgemm_blockedzig)) |
| `vector/gemm_packed.zig` | load-time packed dense f32 NT microkernel and wide-output task splitter ([§9.5](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#95-gemm-dispatch-precedence-blas-and-the-blocked-packed-kernel-srcbackendnativezig-vectorgemmzig-vectorgemm_blockedzig)) |
| `vector/matmul_quant.zig` | quantized matmul dispatch + row/column parallel splitters (kernels live in `backend/quant/*`) |
| `vector/batched.zig` | batched dense GEMM (reuses the `gemm*Range` kernels per batch) |
| `vector/conv.zig` | causal depthwise/general/grouped 1-D conv, dense conv1d/col2im1d, channel-last conv2d + im2col, `Conv1dDims`/`Conv2dDims` |
| `vector/pool.zig` | channel-last pool2d (max/avg/sum), pool backwards, `upsample2xNearest` |
| `vector/winograd.zig` | F(2×2,3×3) and F(4×4,3×3) transform kernels |

`ParallelConfig` is one field: `pool: ?*thread.Pool = null`. Whether a kernel
splits is decided by the thread-count gates in `common.zig`
(`elementwiseThreadCount`, `matmulThreadCount`, `columnThreadCount`,
`i8ColumnThreadCount`, `batchedThreadCount`, `depthwiseConvThreadCount`,
`generalConvThreadCount`) against the tuned thresholds in `src/parallel.zig`:

| Constant | Value | Meaning |
|---|---|---|
| `vector_max_threads` | `-Dmax-threads` (default 8) | comptime team ceiling and stack-array bound |
| `vector_elementwise_len_threshold` | 256 Ki elements | below this, elementwise/conv kernels stay serial |
| `row_kernel_len_threshold` | `vector_elementwise_len_threshold / 2` | pool gate of the fused row kernels (softmax/norm/loss rows, quantized row passes); the halving is policy in one place |
| `vector_matmul_work_threshold` | 1 Mi (m·n·k) | row-split GEMM gate |
| `attention_work_threshold` | `vector_matmul_work_threshold / 2` | pool gate of the attention kernels (same one-place ratio policy) |
| `vector_batched_work_threshold` | 2 Mi | batched GEMM gate |
| `vector_column_min_m` / `vector_column_min_n` | 32 / 128 | column splits are chosen for decode-shaped GEMMs with `m <` the m constant **and** `n ≥` the n constant; at `m ≥ 32` splitting is by rows |
| `vector_column_chunk` | 64 | columns per task in column splits |
| `vector_column_work_multiplier` | 1 | scales the column-split work gate: `columnThreadCount` stays serial below `multiplier × vector_matmul_work_threshold` m·n·k |
| `backward_matmul_work_threshold` | 262 144 | autograd-side pool-enable gate ([§5](05-automatic-differentiation.md)) |
| `backward_async_work_threshold` | 256 Mi | dot-backward async offload gate ([§5](05-automatic-differentiation.md)) |
| `bmm_loop_work_threshold` | 262 144 (= `backward_matmul_work_threshold`) | total m·n·k·batches above which a multi-batch matmul loop splits batches across the pool (`src/exec/matmul.zig`) |
| `bmm_loop_max_chunks` | 16 | chunk cap and stack task-array bound for that batched-loop split |

Parallel splits are deterministic: tasks own disjoint output ranges, so the
threaded result is bit-identical to the serial path for elementwise, conv,
pool, and Winograd kernels (reductions and GEMM state their reassociation
tolerance instead — see [§9.3](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#93-the-scalar-backend-and-the-parity-contract-srcbackendcpuzig-srcbackendparity_testzig)).

Above the backend seam, the exec tier builds streaming inner-lane kernels
from the same primitives (`src/exec/row_ops.zig`): softmax, logsumexp,
logSoftmax, and softmax backward on non-last axes, and layerNorm
backward's inner>1 arm, stream every pass row-major and run `@Vector`
lanes across the contiguous inner index, with pooled scratch rows for the
per-lane max/sum (`vexpf` matches the last-axis row kernels' arithmetic).
The softmax-family kernels split their lane range across the pool — lanes
are independent and scratch columns disjoint, so any task count is
bitwise identical to the serial call (64-lane minimum per task);
layerNorm backward stays serial by design, its dweight/dbias accumulation
crosses lanes.

The primitive vocabulary is also re-exported on the public facade as
`fucina.simd` (`Vf32`, `vector_len`, `vexpf`, `sigmoidVec`, `tanhVec`):
an elemental op ([§4.4](04-tensor-operations.md#44-unary-ops-srcagtensorzig-srcbackendopszig)) may declare vector twins of its scalar rules —
`forwardVec`/`backwardVec` for unary ops,
`forwardVec`/`backwardAVec`/`backwardBVec` for binary — and the elemental
kernels then run vector lanes with the scalar rules on the tail, the same
split as the built-in SIMD kernels. Ops without vector decls are
untouched.

Elementwise entry points operate on `f32` tensors by default; the `*Typed`
twins (`elementwiseContiguousIntoTyped`, `sumSliceTyped`, `dotIntoTyped`,
`matmul2DIntoUncheckedTyped`) accept
`.f16`, `.bf16`, and `.f64` with the compute/output dtype policy from [§8](08-data-types-storage-and-the-raw-tensor-layer-internal.md)
(f16/bf16 accumulate sums and dots in f32).

## 9.5 GEMM: dispatch precedence, BLAS, and the blocked packed kernel (`src/backend/native.zig`, `vector/gemm.zig`, `vector/gemm_blocked.zig`)

Every dense f32 GEMM entry in the native backend dispatches in a fixed order,
each tier compiled in only when its build flag is set:

1. **GPU** (`-Dgpu≠none`): if `gpu.shouldUseGpu(m, n, k)` passes and the
   provider's `gemmF32` returns `true`, done. A `false` return (gate refusal,
   init failure, kill switch, driver error) falls through — correctness never
   depends on the GPU.
2. **BLAS** (`-Dblas≠none`): if `shouldUseBlas(m, n, k)` — all of `m, n, k ≥
   16` and each dimension fits in `c_int` — the call goes to `cblas_sgemm`
   (row-major, `alpha = 1`, `beta = 0`, overwrite). Batched entries require
   `batch_count > 1` on top and loop `cblas_sgemm` per matrix.
3. **Pure-Zig vector GEMM** otherwise.

BLAS providers are linked per `-Dblas`; on first GEMM the native backend pins
the provider's thread count to `-Dblas-threads` (when nonzero) exactly once,
under a mutex, via the provider-specific setter (`openblas_set_num_threads`,
`bli_thread_set_num_threads`, `mkl_set_num_threads`,
`nvpl_blas_set_num_threads`; Accelerate and the generic `blas` provider have
no setter and are left alone).

The accumulate twin `matmul2DAccIntoUnchecked` (`C += A·B`, the backend
seam under `addDot`, [§4.8](04-tensor-operations.md#48-dot-tag-directed-contraction-srcagtensorzig-srctag_opszig)) skips the GPU tier: `shouldUseBlas` winners run
`cblas_sgemm` with `beta = 1` over the addend held in `out`; otherwise the
vector NN family runs its comptime `StoreMode` (`store`/`accumulate`)
kernels, with the blocked path seeding its first k-panel in accumulate
mode (`gemmBlockedAcc`).

The BLAS tier also exposes one raw entry for exec-layer fused kernels:
`sgemmStrided` (`native.zig`) is row-major
`C = alpha·op(A)·op(B) + beta·C` with every leading dimension explicit,
compiled on BLAS builds only (callers comptime-gate on
`backend.native_uses_blas`). The attention-backward BLAS-strip route
issues its per-tile contractions through it (`src/exec/attention.zig`;
`FUCINA_ATTN_BWD_BLAS=0` reverts to the register-tiled route, the only
route on non-BLAS builds).

Within the pure-Zig tier there are two paths:

- **Register-tiled row kernels** (`gemmNNRange`/`gemmTNRange`/`gemmNTRange`):
  loop orders chosen so the inner loop is contiguous streams (NN/TN
  broadcast-FMA into C rows; NT is a per-element two-stream dot). Parallel
  over row ranges, or over column ranges for decode-shaped `m < 32`.
- **The BLIS-style blocked packed GEMM** (`gemm_blocked.zig`) — the
  `-Dblas=none` answer to training-shaped sizes. Gate:
  `shouldUseBlocked(m, n, k)` = `m ≥ 32 and n ≥ 32 and k ≥ 16` and
  `m·n·k ≥ 192 Mi` (`blocked_work_threshold`). Classic three-level loop nest
  `jc(nc) → pc(kc) → ic(mc)` with packed `A~`/`B~` panels and an `mr×nr`
  register microkernel; TransA/TransB are absorbed by the pack loops so all
  three orientations share the microkernel. Comptime microkernel shape:
  `mr = 8, nr = 12` on aarch64 (24 four-wide accumulators), `mr = 6,
  nr = 2·vector_len` elsewhere. Default `BlockParams`: `kc = 128` (aarch64) /
  `512` (x86, `x86_default_kc`), `mc = 128`, `nc = 1024`, bounded by
  `kc_max = 512`, `mc_max = 256`, `nc_max = 1024`;
  `gemmBlockedWithParams` **panics** on out-of-bounds params (they would
  overrun the static workspace, and the bench sweep feeds runtime params in
  ReleaseFast where asserts vanish). Because backend kernels must stay
  allocation-free and these entries are infallible, the pack panels live in a
  static BSS workspace guarded by `workspace_lock`; concurrent blocked GEMMs
  serialize on it (an accepted trade — pool workers never re-enter the path).
  Parallelism is an (ic-block × column-chunk) cell grid over the persistent
  team; each C tile is written by exactly one task, so results are
  deterministic and thread-count-independent.

The public load-time dense packed operation is a separate NT path for reused
weights. `packRhs` snapshots an f32/f16/bf16 `[n,k]` weight into owned f32
output-row panels (logical rows remain contiguous; `n` is padded to four), and
`dotPacked` computes `A[m,k] · W[n,k]ᵀ`. Its fixed dispatch table is:

| Build/cell | Packed-dense dispatch |
|---|---|
| eligible GPU | provider first, using the stable packed tensor |
| BLAS build, `m,n,k ≥ 16` | existing `shouldUseBlas` winner — except the band below |
| Accelerate build, `m < 32, k ≥ 4096` | packed microkernel (with the RHS already packed it beats BLAS on skinny-m tall-k cells; scoped to Accelerate — the OpenBLAS/x86 crossover is unmeasured) |
| BLAS build, any dimension `< 16` | packed microkernel |
| `-Dblas=none` | packed microkernel |
| `-Dbackend=scalar` | scalar packed-op specification |

The microkernel (`vector/gemm_packed.zig`) keeps four RHS output rows and three
AVX2 (six aarch64) input rows live in registers, with SIMD lanes along `k`.
It splits work over four-column panels in the wide output dimension and emits
three dynamically claimed tasks per participant, so skinny `m` does not limit
the team to `m` jobs. Packing and output allocation remain outside the compute
leaf. Ordinary `dot`/`matmul` dispatch, large-`m` blocked/BLAS winners, and GPU
precedence are unchanged.

f16 GEMM policy (`vector/gemm.zig`): on aarch64 the f16×f16 `@mulAdd` arms
are native `fmla.8h`, so half-precision accumulation is the fast path and
output is bit-stable across releases; every other ISA takes widened twins
(each f16 load converted once, f32 accumulation — strictly more accurate, and
different from the aarch64 bit pattern). `matmulTransB2DIntoUncheckedBf16Rhs`
dots f32 activations against a bf16 RHS without materializing f32 weights.

Batched GEMM (`vector/batched.zig`) parallelizes across batches
(`batchedThreadCount`) and reuses the row-kernel ranges per matrix; `stride_b
= 0` broadcasts one RHS across the batch and `stride_a = 0` shares one LHS.

Threshold provenance and the re-tuning protocol (cool-state runs, thermal
discipline, paired A/B) live in [BENCHMARK.md](../BENCHMARK.md); the sweep tool
is `zig build bench-gemm` (`-Dblas=none -- --sweep` for block params,
`-Dgpu=metal` for GPU crossover points).

## 9.6 Convolution, pooling, and image kernels (`vector/conv.zig`, `vector/pool.zig`, `vector/winograd.zig`)

All 2-D image kernels are channel-last: input `[H, W, Cin]`, weights
`[Cout, KH, KW, Cin]`, output `[OH, OW, Cout]`, so every window step is a
contiguous `C`-wide vector op. Forward kernels parallelize over output rows
(bit-identical to serial); the conv2d backward kernels split the same way
(input rows for backward-input, output channels for backward-weight — each
task owns disjoint output, so parallel is bit-identical to serial), while
the pool2d backward scatters are correctness-first serial. Pooling
semantics: `.max` skips out-of-range taps (−inf border, the
ONNX convention), `.avg` averages over *valid* taps only
(`count_include_pad=0`), `.sum` sums valid taps (the upsample VJP; not
exposed publicly). The 1-D families cover causal depthwise FIR
(DeltaNet-style, `[time, channel]` × `[channel, tap]`), channel-mixing dilated
causal conv (`[tap, in, out]` weights so every kernel runs on contiguous
out-channel rows), grouped causal conv, and general non-causal conv1d with
`col2im1d` for transposed conv.

How a conv2d reaches these kernels is exec-side routing
(`src/exec/conv.zig`), but its gates are backend facts worth stating here.
For the dominant shape — 3×3, stride 1, `pad ≤ 1`, `groups == 1`,
`cin ≥ 4`, `oh, ow ≥ 2` — the op takes the **Winograd route**: weight
transform `U = G·g·Gᵀ`, input transform `V = Bᵀ·d·B`, per-plane tile GEMMs
through the ordinary matmul dispatch (BLAS / blocked / row kernels), output
transform with bias folded in and an optional fused ReLU epilogue. Two tiers:

- **F(2×2,3×3)**: 16 coefficient planes, ~2.25× fewer MACs than im2col,
  ~1e-6-relative drift vs the direct kernel (reassociated 3×3 reduction).
- **F(4×4,3×3)**: 36 planes, ~4× fewer MACs, ~1e-5-relative drift; selected
  for large shallow maps — `min(oh, ow) ≥ 14` and `cin ≤ 56` by default
  (bench-tuned: detector-class maps win ~22%, deep-channel recognizer stacks
  lose ~30% and stay on F2).

Gating is read once and cached: the route defaults **on for `-Dblas=none`**
builds and **off when a platform BLAS backs the matmul**
(`tuning.Table.winograd`, default `!use_blas` — Accelerate's AMX prefers one
big im2col GEMM). Runtime overrides: `FUCINA_WINOGRAD=1` forces on, `=0`
forces off, `FUCINA_WINOGRAD_F4=0` pins large maps to F2,
`FUCINA_WINOGRAD_F4_MIN` / `FUCINA_WINOGRAD_F4_MAXCIN` retune the F4
shape gate. Ineligible convs fall to im2col + one GEMM (`groups == 1`) or the
direct grouped kernel; 1×1 stride-1 pad-0 convs lower to a plain NT matmul.

## 9.7 Quantized matmul dispatch, packed RHS, and the int8 dot arms

The quantized *kernels* belong to [§10](10-quantization.md); this section covers what the backend
tier owns: dispatch, scratch, and ISA selection.

`matmul2DQuantizedRhs` switches over `AnyQuantizedMatmulRhs` (one
arm per GGML format) and, per call, quantizes the f32 activation rows into
the format's activation blocks — `Q8_0`/`Q8_1` for legacy and
`IQ4_NL`/`MXFP4`/`NVFP4` formats, `Q8_K` for K-quants and the `IQ*`/`TQ*`
table formats — using the caller's allocator (the deliberate scratch tier
from [§9.2](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#92-the-kernel-interface-and-the-kernel-contract-srcbackendinterfacezig-srcbackendzig)). The x4/x8 interleaved fast paths add row-shape policy, tuned so
every row's math stays bit-identical to the kernel that owns it:

- `Q8_0x4`: `m % 4 == 0` goes straight to the packed kernel; `12 ≤ m < 32`
  uses a padded-LHS variant; `m ≥ 32` splits into a multiple-of-4 bulk (x4
  kernel) plus a 1–3-row remainder (row kernel); small odd `m` takes the
  per-row path.
- `Q4_Kx8` engages the x4 activation packing for every `m ≥ 4`
  (`q4_k_x4_min_rows`; its padded-group kernel makes one pass over the packed
  weights); `Q5_Kx8` has no padded kernel, so its bulk+tail split only pays
  at `m ≥ 128` (`q5_k_x4_prefix_min_rows`).
- `Q4_Kx2Mmla` (aarch64 `smmla`) processes row pairs with a `Q8_Kx2` LHS
  packing and a row-kernel tail.

The `matmulPacked*Slice` entries are the same kernels with a
**pre-quantized LHS** — exec's fused split-activation FFN paths quantize
activation rows themselves, so these skip the allocator tier entirely.

**Packed RHS, user-facing story.** Packing a weight once at load time and
matmul-ing against the pack is the hot inference path. The public facade has
two layout families, plus an older backend-only typed bridge:

- Dense f32/f16/bf16 weights: `weights.packRhs(ctx)` snapshots a rank-2
  `[out, contract]` weight into an owned `fucina.PackedRhs(dtype)`, the
  `PackedDenseRhs` f32 panel (`dtype == .f32`). f16/bf16 values widen once
  at pack time. The public
  `x.dotPacked(ctx, &packed, contract_tag, out_tag)` takes an f32 lhs and
  dispatches through GPU, BLAS, or the skinny-m wide-output microkernel by the
  fixed table in [§9.5](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#95-gemm-dispatch-precedence-blas-and-the-blocked-packed-kernel-srcbackendnativezig-vectorgemmzig-vectorgemm_blockedzig). The source can be released after packing; the model
  owns and deinitializes the pack.

- Block-quantized weights: the container types `QuantizedMatmulRhsQ8_0x4`,
  `Q6_Kx4`, `Q4_Kx4`, `Q4_Kx8`, `Q4_Kx2Mmla`, and `Q5_Kx8` are the
  interleaved layouts, each carrying `pub const dtype`;
  `backend.PackedRhsFor(dtype)` (the facade's `fucina.PackedRhs(dtype)`) is
  the one dtype-to-container map and picks the ISA-best container per dtype
  (`q8_0→x4`, `q6_k→x4`, `q5_k→x8`, `q4_k→x2mmla` when
  `supports_q4_k_mmla` else `x8`). Model code calls
  `weights.packRhs(ctx)` / `packRhsAs(ctx, fucina.quant.QuantizedMatmulRhsQ4_Kx8)`
  on a rank-2 quantized tensor and feeds the pack to `dotPacked` — full
  semantics in [§10](10-quantization.md).
- Backend-only same-dtype f16/bf16 bridge: `backend/packed.zig` also defines
  `PackedMatmulRhsFor(dtype)` for `.f16`/`.bf16` (the container's `dtype` is
  the 16-bit source); the pack widens the RHS to f32 once (`packRhs` → owns
  an f32 tensor; caller `deinit()`s the container).
  `matmul2DIntoUncheckedPackedRhsTyped` then runs f32 GEMM with widen/narrow
  bridges, with a dedicated `m == 1` GEMV fast path that dots the f16/bf16
  activation row directly against the packed f32 columns (column-parallel
  over the pool). This bridge preserves a same-dtype f16/bf16 result and is
  distinct from the public f32-lhs `PackedDenseRhs` panel. It is reached
  through `ExecContext.packMatmulRhs` (`.f16`/`.bf16`) /
  `matmulPacked` ([§6](06-the-execution-runtime-execcontext-and-the-memory-model.md)).

**Arch-gated int8 dot arms.** The K-quant/Q8 kernels select their inner dot
at comptime: aarch64 `sdot` inline asm (all aarch64), aarch64 `smmla` behind
`has_aarch64_i8mm` (Graviton3+/Grace-class; selects the `Q4_Kx2Mmla`
layout), x86 AVX2 via the `vpmaddubsw`+`vpsignb` sign trick, AVX-VNNI via the
VEX-encoded `vpdpbusd` (the `{vex}` prefix is mandatory — LLVM's asm parser
does not feature-check and a bare `vpdpbusd` assembles to the EVEX form and
SIGILLs on Alder/Raptor Lake), AVX512-VNNI via EVEX `vpdpbusd`, and a
portable widening tier everywhere else. Debug builds run the portable twins
(the stage2-assembler gate, `common.has_llvm_asm`) — ISA-arm coverage
requires ReleaseFast/ReleaseSafe.

`src/x86dot_check.zig` is the standalone cross-ISA parity checker for these
arms plus the Q4_K/Q8_0/TQ2_0 dot kernels: a self-contained `main` that runs
kernel-vs-scalar semantic asserts on deterministic randomized and extreme
inputs, exits nonzero on mismatch, and prints FNV-1a checksums of the raw
result bits so runs from different machines/emulators can be diffed for
bit-exactness. `zig build x86dot-check` builds and runs it natively
(ReleaseSafe) and additionally compile-checks — never runs — one leg per
feature gate no local substrate can execute (`x86_64_v3`, `alderlake`,
`znver4`, `neoverse_v1`). The file's header carries the dated per-arm
execution attestation table (which arms have actually executed on which
hardware/emulator) and the emulator caveats (some emulators
executes AVX2 silently wrong).

## 9.8 Threading: the worker team (`src/thread.zig`, `src/parallel.zig`)

The team itself is context infrastructure and is documented in [§6.6](06-the-execution-runtime-execcontext-and-the-memory-model.md#66-the-worker-team-srcthreadzig-srcparallelzig): pool
creation and sizing (`cpuThreadCount`, `setMaxThreads`,
`FUCINA_MAX_THREADS`, `FUCINA_SPIN_BUDGET`), the spin-then-park worker
lifecycle, and the `parallelChunks`/`parallelChained` dispatch contracts.
This subsection covers the backend seam.

**The cross-thread handshake.** The pool is created lazily:
`tryWorkPool` (under `work_pool_mutex`) initializes one `thread.Pool`
per `ExecContext` with `cpuThreadCount(vector_max_threads) - 1` workers and
publishes it in `ExecContext.parallel_pool` — the atomic
release-store/acquire-load pair from [§6.6](06-the-execution-runtime-execcontext-and-the-memory-model.md#66-the-worker-team-srcthreadzig-srcparallelzig) and [§9.2](09-backends-cpu-simd-blas-threading-and-gpu-offload.md#92-the-kernel-interface-and-the-kernel-contract-srcbackendinterfacezig-srcbackendzig), because a kernel
dispatched on another thread may race the publication; every kernel call
reads it through `ctx.pc()`. `ExecContext.deinit` unpublishes (stores
`null`) before destroying the pool. Exec ops call
`enableNative*PoolForWork(work, threshold)` helpers so the team is only
instantiated once an op is actually big enough to split. `thread.zig` also
provides `Mutex`/`Condition` (thin `std.Io` wrappers), `WaitGroup`,
`ThreadSafeAllocator` (mutex-guarded child allocator), `Chain`, and
`OneShotWorker` — a single persistent futex-parked thread used by
dot-backward to overlap the two gradient GEMMs ([§5](05-automatic-differentiation.md)).

## 9.9 GPU offload (`src/backend/gpu.zig`, `metal.zig`, `cuda.zig`)

Both providers implement the same eager accelerator contract:

- **Gates decide, dispatchers run.** Cheap `shouldUse*` gates (no device
  init) sit at the dispatch sites; every dispatch entry returns
  `false`/`null` when the GPU did not run, and the caller falls through to
  BLAS/vector — correctness never depends on the GPU.
- **Submit eagerly; synchronize at host visibility.** Dense f32 commands are
  encoded and submitted before the op returns. Output storage carries only a
  completion/lifetime token: GPU consumers remain queue-ordered and a CPU
  data accessor/kernel performs the deferred wait. F16 and quantized panel
  staging remain blocking. The public `Tensor` API still has no device type or
  location state, and no op description/graph is retained.
- **Lazy init.** The device/library/context is created on first
  above-threshold use, double-checked under a mutex so concurrent
  `ExecContext`s share one device. Config (env vars) is read once, separately
  from device init, so below-threshold probes stay cheap.
- **Shape gates.** General GEMM requires `m ≥ 32, n ≥ 32, k ≥ 16`.
  Resident dense-f32 GEMV/small-m GEMM has the separate `m ≤ 8`,
  `n,k ≥ 256`, `FUCINA_GPU_MIN_WORK_GEMV` gate.

The `FUCINA_GPU*` runtime knobs — kill switch, per-kind work gates, the
MoE fill gate, tracing, TF32, the transient floor, VRAM budget, kernel
source, and the decode opt-in — are all read once at first use and are
tabulated with their defaults in [§2.6](02-toolchain-build-and-project-wiring.md#26-runtime-environment-variables).

The eager f32/f16/dense-quant implementation and its ordering/teardown contract are described
in [GPU-OFFLOAD.md](../GPU-OFFLOAD.md). Both providers keep their queue/streams
and library state open across calls. Metal additionally caches storage page
wrappers; CUDA uses a bounded eight-slot device pool, registers pooled host
allocations once, and connects persistent upload/compute/download lanes with
events. A pending producer's device address passes directly to a dependent
GEMM.

### 9.9.1 Metal (`src/backend/metal.zig`, `-Dgpu=metal`, macOS)

The kernels are vendored MSL compiled once at lazy init from embedded source
by the ObjC shim (`src/backend/metal/shim.m`): the MLX "steel" f32/f16 GEMM
(`metal/mlx_gemm.metal`, MIT, Apple) and the llama.cpp quantized `mul_mm`
(`metal/ggml_mul_mm.metal`, dequant-in-kernel). What offloads:

- **Dense f32 GEMM/GEMV** `nn`/`tn`/`nt` and strided-batched
  (`gemmF32Async`/`gemmBatchedF32Async`; a batched call is ONE dispatch with
  grid depth = batch), behind the general or resident-small-m gate. The direct
  slice `gemmF32` twin is blocking for parity/benchmarks.
- **f16 NT GEMM** (`gemmF16NtAsync`) behind `shouldUseGpuF16ForRhs`: the
  mixed steel instantiation reads f16 operands and writes the public f32
  output directly. It commits immediately and attaches a Work; there is no
  shared staging buffer, f16 lock, CPU widen pass, or result re-rounding. The
  old direct-slice `gemmF16Nt` remains blocking only for low-level parity
  tests/bench callers.
- **Dense quantized prefill** (Q4_K/Q6_K/Q8_0/TQ2_0): exec's
  `denseQuantMatmulGpu` seam (`src/exec/quant_matmul.zig`) offloads
  `m ≥ 32` stable-weight matmuls behind the compact/raw or packed-CPU
  per-format gate when
  `k % KernelFormatTag.kMultiple() == 0` (32 for q8_0, 256 for
  q4_k/q6_k/tq2_0; the tag comes from the provider's `kernelTag(dtype)`) and
  `n % 4 == 0`. `gemmQuantNtAsync` binds input/output tensor storage
  directly and copies its ≤4 KiB tile table into command-owned bytes (up to
  8192 rows); shared-input batches encode multiple weight matrices without
  replicating input rows. Transient RHS or longer prompts retain the blocking
  chunk fallback.
- **PTQTP ternary prefill** (`src/weights.zig`): with resident plane
  bytes and `m ≥ 32`, each TQ2_0 trit-plane runs as one dequant-in-kernel
  dispatch through the same `denseQuantMatmulGpu` seam and the K plane
  outputs sum on the CPU (K = 1 returns the async tensor directly). A
  tie-fitted K = 2 pair whose folded pack is resident takes
  `foldedTernaryMatmulGpu` instead — ONE `fucina_mul_mm_tq2_0_folded_f32`
  dispatch, async return, no plane sum — behind the provider capability
  `has_tq2_0_folded_quant`. Both arms share the ternary work gate
  (`FUCINA_GPU_MIN_WORK_DENSE_TQ2`, [§2.6](02-toolchain-build-and-project-wiring.md#26-runtime-environment-variables)); any refusal falls through to
  the x4 interleaved CPU kernels wholesale. Not bitwise vs the CPU chain —
  the same accepted numerics stance as the q4_k/q6_k/q8_0 dense offload.
- **MoE expert FFN** (models tier, `src/models/gemma/moe.zig`): CPU gathers
  activation rows into shared staging panels (`qmoeStage`, grow-only
  MTLBuffers), dispatches grouped tile-table GEMMs (`gemmQGroupedNt`, one
  `QMMTile` per 32-row output tile per expert), and reads results back — all
  under the process-global `qmoe_lock`. Gated by `shouldUseGpuQMoe` (total
  m·n·k across both projections) *and* `qmoeFillAcceptable` (per-tile GPU
  cost is fill-independent, so below ~50% occupancy the CPU wins).

Weight residency: `allocResidentBytes(len)` returns device-owned,
page-aligned unified-memory bytes the CPU reads through the same slice; this
is a performance cache, not a correctness precondition — pageable client
wraps are re-wired into the GPU address space on every commit (~45 µs/MB), so
stable weights should live resident. The provider keeps a bounded
address-keyed registry (512 ranges) so dispatch paths recognize resident
operands without caller flags. Dense f32/f16 inputs and f32 outputs own one
page wrapper on each storage allocation; pooled reuse changes values, not the
mapping, and the wrapper is evicted before the allocation is freed. Quantized
stable weight pages retain the bounded address-keyed shim cache. The stale-pages rule
is absolute: only bytes whose address is process-lifetime-stable may be
flagged cacheable
(`RhsLifetime.stable_process`) — a cached wrap of a freed-and-reused page
reads stale data. `freeResidentBytes` unregisters and releases.

What does **not** offload on Metal, per current tree: **quantized** decode GEMV
(`shouldUseGpuQuantDecode` is hard-`false`; resident dense-f32 GEMV is separate),
attention (`shouldUseGpuAttn`/`attnPrefillF16` return
`false`; the CPU tiled kernel runs), and the ES parameter-update device arm
(`flatPerturb`/`flatWeightedUpdate`/`flatAnchorDecay` are stubs returning `false` — on unified
memory the CPU kernels already mutate the shared pages the GPU reads
zero-copy). Quantized matmuls reached from *trainable* autograd inputs pass
`allow_gpu = false` (`QuantizedMatmulOptions`), keeping the training path on
CPU unless a gradient-aware GPU policy is added deliberately.

### 9.9.2 CUDA (`src/backend/cuda.zig`, `-Dgpu=cuda`, Linux)

Host binding is `dlopen` (`src/backend/cuda/api.zig`): `libcuda.so.1` and
`libcublas` are loaded at runtime, so **no CUDA SDK is needed at build time**
and `-Dgpu=cuda -Dtarget=x86_64-linux-gnu` cross-compiles from any machine.
Missing libraries degrade per capability (no cuBLAS ⇒ only the f32/f16 GEMM
arms are disabled). Quantized/GEMV/ES/attention kernels are vendored CUDA C
(`cuda/kernels.cu`) shipped as committed NVRTC-generated PTX
(`cuda/kernels.ptx`, driver JIT, disk-cached, ~26 ms cold), with an NVRTC
recompile fallback (`FUCINA_GPU_KERNELS=src` forces it). Persistent upload,
compute, and download streams use reusable events; cuBLAS stays bound to
compute and submission holds a short dispatch lock. Pooled f32 host allocations are registered once for
direct asynchronous DMA. What offloads:

- **f32 GEMM/GEMV** nn/tn/nt + strided-batched via cuBLAS, strict FP32 math by
  default (`FUCINA_GPU_TF32=1` opts into TF32). The f32 gates add a
  *transient floor* on top of `min_work`: non-resident operands stream over
  PCIe (measured ~10.6 GB/s pageable), so shapes below `2^33` m·n·k or
  `m < 128` are refused even when the plain gate passes (trace counts these
  separately as re-tuning evidence). An already device-resident RHS instead
  uses the `2^27` resident gate; resident `m≤8` uses the separate GEMV gate.
  Pending outputs pass their device address directly to a dependent op. H2D,
  compute, and D2H overlap on their three lanes; the final D2H lands directly
  in the ordinary exec-owned output (a pinned-stage fallback remains for a
  host allocator the driver cannot register).
- **f16 NT GEMM** via `cublasGemmEx` (f16 operands, direct f32 output and f32
  accumulation) through the same eight async slots and three persistent lanes
  as f32. Resident RHS decode uses the separate `2^20` gate; transient decode
  is refused, while streamed prefill retains the `2^27` gate.
- **Dense quantized prefill** (Q4_K/Q5_K/Q6_K/Q8_0; Q5_K is CUDA-only): stable RHS bytes resolve to one
  managed resident allocation. `gemmQuantNtAsync` reuses the slot's activation,
  output, pinned-tile, and device-tile buffers; a pending f32 producer passes
  its device address directly. Shared-input batches launch each weight matrix
  on the same stream without copying activation rows. On tensor-core devices,
  adaptive N32/N64 WMMA kernels consume the same half-rounded dequantized
  operands as the scalar fallback and accumulate f32; N32 is retained for
  severe grouped/unsplit grid underfill. If the dense N64 grid fills less than
  roughly 7/8 of the SMs, prefill splits K two ways (up to three for Q6_K)
  into a reusable per-slot buffer and queues a deterministic reduction on the
  same stream before the completion event. No host synchronization is added.
  Full output tiles store directly and partial tiles use guarded staging.
  `FUCINA_GPU_QUANT_MMA=0` retains the scalar path;
  `FUCINA_GPU_QUANT_SPLIT_K=0` isolates unsplit WMMA.
  Host download is deferred to the output Work.
- **Grouped MoE** uses the same tile kernel but keeps its required CPU phase
  boundaries (`qmoeStage`, `qmoe_lock`). Panel/tile H2D, compute, and panel D2H
  are now event-chained across persistent streams; the CPU performs one final
  fence when GeGLU/scatter needs the result, rather than synchronizing compute
  and then starting a blocking download. The provider-level Q5_K grouped
  kernel is parity-tested, but current model-specific MoE loaders do not route
  Q5_K expert stacks to it; dense Q5_K linears are wired end to end.
- **Fused prefill attention** (`attnPrefillF16`): online-softmax grouped
  attention over f16 KV with the CPU tiled kernel's exact semantics
  (absolute positions, pre-clamped sliding window, causal or bidirectional,
  per-head KV mapping, `d ≤ 256`). Stateless and blocking — Q/K/V stream in,
  the output streams back. Gated by `shouldUseGpuAttn` on q·kv·heads·d ≥
  `2^28`; decode never reaches this seam, and the KV cache itself stays in
  host memory.
- **Quantized decode** (opt-in, `FUCINA_GPU_DECODE=1`) against
  **resident-or-adoptable weights only**: Q4_K/Q6_K/Q8_0 use warp-per-row
  dequant-dot for `m ≤ 8`; Q5_K uses that compact-row route for `m < 4` and
  the tensor-core tile route for `m = 4..8`. At decode
  shapes the op is bytes-bound and streaming weights per token is a strict
  loss, so a registry miss on transient RHS refuses and the caller stays on
  the CPU int8 kernels. Q5_K additionally applies
  `FUCINA_GPU_MIN_WORK_DECODE_Q5`: 1×4096² stays on CPU, while the measured
  1×6144×4096 and rows 2–8 cross to CUDA. The global decode arm remains
  opt-in even though a Q5_K_M 32-token greedy CPU/CUDA oracle matched exactly.
- **ES device arm** (`flatPerturb`, `flatWeightedUpdate`, `flatAnchorDecay`): seeded
  perturbation/update/anchor kernels for evolution-strategies training ([§11](11-training-optimizers-evolution-strategies-lora-and-checkpoints.md))
  that write the caller's live resident storage (never an adopted snapshot)
  and reproduce the CPU noise contract bitwise.

Residency: `allocResidentBytes` = `cuMemAllocManaged` + `READ_MOSTLY` advice
+ prefetch-on-first-use — unified addressing means a resident RHS dispatches
with zero weight transfer while the CPU fallback reads the same pointer.
Stable (`RhsLifetime.stable_process`) RHS bytes are additionally *adopted*
into the managed registry on first use — the analog of Metal's wrap cache,
same stale-pages rule: mmap'd weights cross PCIe once per process, not per
dispatch. `FUCINA_GPU_VRAM_BUDGET` bounds tracked allocations (default ~80%
of free VRAM at init); over-budget requests return `null` and callers fall
back to host bytes + transient (the Metal OOM path). Residency requires
`CONCURRENT_MANAGED_ACCESS` (absent on WSL2/some Jetson targets → residency
disabled, transient/CPU fallback with a one-time warning).

### 9.9.3 `internal.gpu` hooks and tracing (`src/fucina.zig`)

Users keep ordinary eager `Tensor` values; residency and tracing are
backend-owned details deliberately kept off the public root, under
`fucina.internal.gpu`:

```zig
pub const gpu = struct {
    pub const enabled = backend.gpu_impl.enabled;             // comptime: -Dgpu build?
    pub const has_quant_gemm = backend.gpu_impl.has_quant_gemm; // dequant-in-kernel GEMM?
    pub const has_q5_k_quant = backend.gpu_impl.has_q5_k_quant; // CUDA Q5_K capability?
    pub const has_tq2_0_quant = backend.gpu_impl.has_tq2_0_quant; // ternary TQ2_0 kernel (Metal)
    pub const allocResidentBytes = backend.gpu_impl.allocResidentBytes;
    pub const freeResidentBytes = backend.gpu_impl.freeResidentBytes;
    pub const traceEnabled = backend.gpu_impl.traceEnabled;
    pub const traceReset = backend.gpu_impl.traceReset;
    pub const traceDump = backend.gpu_impl.traceDump;
};
```

`has_quant_gemm` is the capability loaders key on when reshaping CPU-side
weight representations for the GPU quant path (e.g. `src/models/gemma/model.zig`
copying mmap'd expert tensors into resident storage) — a provider can be
`enabled` while its quantized arms are stubs. `RhsLifetime`
(`fucina.RhsLifetime`) is how callers communicate the storage-stability
guarantee that authorizes address-keyed caching of an RHS; the enum, the
cacheability rule, and the pooled-storage prohibition are [§6.7](06-the-execution-runtime-execcontext-and-the-memory-model.md#67-rhslifetime-address-keyed-caching-of-rhs-operands-srcexecquant_matmulzig).

Tracing workflow: run with `FUCINA_GPU_TRACE=1`, warm the workload, call
`traceReset()` at the start of the measurement window and `traceDump()` at
the end. The dump (stderr) breaks down per-kind dispatch counts and wall/GPU/
scheduling time (f32/f16/quant, plus gemv/attn and H2D/D2H bytes on CUDA),
resident-vs-streamed RHS counts, resident-bytes allocations, and — most
useful for threshold tuning — the gate-decision counters (pass, below-gate,
shape-reject, transient-floor, shim/CUDA errors) and the top shapes by
dispatch wall time (Metal). `traceReset`/`traceDump` are no-ops when tracing
is off, so instrumentation can stay in place unconditionally:

```zig
fn snippetGpuTraceAndResidency(weight_bytes: []const u8) void {
    const gpu = fucina.internal.gpu;
    // The trace hooks are callable on every build; they no-op unless the
    // process runs with FUCINA_GPU_TRACE=1 on a -Dgpu build.
    gpu.traceReset();
    // ... run the measured (warm) workload ...
    gpu.traceDump(); // per-kind dispatch counts/wall/GPU time to stderr

    if (comptime gpu.enabled) {
        // Device-owned weight storage: stable bytes dispatch with zero
        // per-call transfer; release through the same hook.
        if (gpu.allocResidentBytes(weight_bytes.len)) |dev| {
            @memcpy(dev, weight_bytes);
            // ... hand `dev` to the model; when the owner drops it:
            gpu.freeResidentBytes(dev);
        }
    }
} // requires a -Dgpu build to actually offload
```

GPU parity is tested in-tree: the provider modules carry their own test
blocks (f32 orientation/edge-tile parity vs a f64 reference, f16 NT, the
quantized formats vs dequantized references, grouped expert tiles, the fill
gate arithmetic), compiled and run only on `-Dgpu` builds — `zig build test
-Dgpu=metal` on an Apple Silicon machine, `-Dgpu=cuda` on a CUDA box.
Threshold defaults are measurement-backed; the protocol and the recorded
numbers live in [BENCHMARK.md](../BENCHMARK.md).
