# TERNARY — {-1, 0, +1} weights as a first-class citizen (TQ2_0)

Design record, 2026-07-07. Covers the inference kernels, the encoders, the
straight-through-estimator training op, the ternary-native evolution-strategy
trainer, and GGUF interop. Each claim names the pipeline it covers.

## Why TQ2_0 (and not a new format)

Fucina already carried ggml's ternary types end-to-end (`DType.tq1_0`/`tq2_0`,
GGUF type ids 34/35, cold scalar dot kernels) — decode/matmul only, no
encoder, no fast path, quantized weights frozen under autograd. This work
promotes **TQ2_0** (2.0625 bits/weight: 256-element blocks, `qs[64]` 2-bit
crumbs storing `w+1 ∈ {0,1,2}`, inline fp16 scale `d`) to a first-class
format instead of inventing a Fucina-specific one:

- **GGUF interop is free**: llama.cpp reads/writes the same blocks
  (`--dtype tq2_0` in `export-gguf` emits `general.file_type = 37`).
- The layout is already SIMD-shaped: within each 32-byte group, crumb lane
  `L` covers 32 consecutive activations, so unpack is shift+mask only.
- The BitNet-native alternatives (bitnet.cpp `I2_S`, block 128/64, per-tensor
  scale appended after the bits) differ only in bit order and scale placement;
  measured within ~6% of TQ2_0-class kernels on dot-capable CPUs
  (arXiv:2502.11880 Table: i7-13700H 3.8B, I2_S 35.04 t/s vs TQ2_0 33.19).
- The TL1/TL2 lookup-table kernels (bitnet.cpp) win mainly on CPUs without
  int8 dot instructions and on footprint (TL2: 1.67 bpw); they are GEMV-only
  and need offline per-shape codegen. Deliberately **not** ported; recorded
  as future work. `tq1_0` (1.6875 bpw, base-3^5 packing) stays cold/compat.

## The int8 flagship kernel (inference pipeline)

`src/backend/quant/ternary.zig`. The crumbs multiply as *unsigned codes*:

```
dot(w, a) = Σ (w+1)·a − Σ a
```

Q8_K activation blocks already carry `bsums` (per-16 sums), so `Σ a` is one
vector fold per 256-block and the hot loop is pure shift/mask + int8 group
dots — **no weight multiplications anywhere**:

- aarch64: `sdot` on 16-byte granules (codes {0,1,2} are valid signed bytes).
- x86 AVX-VNNI/AVX512-VNNI: `vpdpbusd` on 32-byte granules (codes are u8).
- x86 AVX2: `vpmaddubsw` + `vpmaddwd(+1)` — a maddubs pair sum is at most
  2·127·2 = 508, so the i16 stage cannot saturate (unlike bitnet.cpp's
  4096-element i16 cadence, which is only statistically safe).
- everywhere else: the portable `@Vector` twins of those primitives.

Every arm accumulates the exact per-block integer (max |Σ(w+1)a| = 65024,
far inside i32), so **all arms are cross-ISA bitwise identical** to the cold
scalar reference — pinned by `ternary_tests.zig` (hot vs cold bitwise) and by
`zig build x86dot-check` (tq2_0 section; aarch64 sdot, x86 AVX2-maddubs, and
x86 AVX-VNNI arms all hardware-executed 2026-07-07 — see the x86 addendum).

Tiles process 4 weight rows per activation pass (`blockCodeDot4`: shared
activation vectors and bsum total), with the standard row/column parallel
split (`vector.matmul2DTQ2_0RhsIntoWithConfig`); `native.zig` routes
`.ggml_tq2_0` matmuls here (scalar backend intentionally keeps the cold
reference path). Constraint: the contract dim k must be a multiple of 256.

**Measured (M1 Max, single thread, ReleaseFast, `zig build bench-ternary`,
2026-07-07):**

| shape | m | cold µs | hot µs | hot/cold | Q4_K µs | dense f32 µs |
|---|---|---|---|---|---|---|
| n=4096 k=4096 | 1 | 1013 | **238** | 4.25x | 525 | 16725 |
| n=4096 k=4096 | 128 | 130767 | **31260** | 4.18x | 69834 | 5983* |
| n=11008 k=4096 | 1 | 2740 | **667** | 4.11x | 1391 | 31527 |
| n=11008 k=4096 | 128 | 352930 | **86815** | 4.07x | 182780 | 8086* |

\* dense f32 goes through Accelerate (multi-core AMX) at those shapes; the
quant kernels are pinned single-thread here by design. The hot kernel is
~2.1x the tuned Q4_K row kernel at equal shapes — the 2.06-vs-4.5 bpw ratio —
and hot-vs-cold checksums match bitwise in ReleaseFast. Scaling is linear in
m at ~3 SIMD ops per 16 weights (shift, mask, sdot): the kernel sits at the
NEON ALU limit (~250 µs theoretical for the 4096² GEMV vs 238 µs measured),
so column-outer re-blocking buys nothing on this target; the multi-thread
split is where prefill throughput comes from.

**x86 addendum (i9-13950HX Raptor Lake, Linux, single thread, ReleaseFast,
`-Dblas=openblas`, 2026-07-07):**

| shape | m | cold µs | hot µs | hot/cold | Q4_K µs |
|---|---|---|---|---|---|
| n=4096 k=4096 | 1 | 985 | **193** | 5.10x | 924 |
| n=4096 k=4096 | 128 | 127171 | **25323** | 5.02x | 123093 |
| n=11008 k=4096 | 1 | 2701 | **534** | 5.06x | 2664 |
| n=11008 k=4096 | 128 | 346312 | **68714** | 5.04x | 345716 |

5.0–5.1x the cold path and ~4.8x Q4_K on this box (`vpdpbusd` consumes 32
bytes/instruction). Hardware attestation: full `zig build test` green
natively (AVX-VNNI arms), `x86dot-check` PASS both natively and with
`-Dcpu=x86_64_v3` (AVX2-maddubs arm, `avxvnni=false`) — the two arms'
checksums are bit-equal (`b1f84dde82d0c0a4`); coverage table in
`src/x86dot_check.zig`. `es-ternary-spirals` end-to-end on the same box:
100% accuracy in 9250 iterations, 112.7 s (vs M1 Max: 100% in 14750
iterations, 104.4 s).

## The mul-free f32 path (training-forward pipeline)

For STE training the forward runs **exact f32 activations** (no activation
quantization) yet still multiplication-free, via the sign-plane/zero-plane
identity — exact in IEEE fp32:

```
w·x = (x XOR s) AND m      s = 0x80000000 where w == −1 else 0
                           m = 0xFFFFFFFF where w != 0 else 0
```

`dotTQ2_0F32` fixes a 4-lane accumulation order, so this path is bitwise
reproducible across every ISA (pinned by an order-matched scalar replica in
tests and x86dot-check). It is a correctness-first path: ~15x slower than the
int8 flagship at GEMV (3.5 ms vs 238 µs on the 4096² shape) but exact.

## Encoders (both pipelines)

- `quantizeRowTQ2_0Into`: ggml `quantize_row_tq2_0_ref` parity — per-block
  absmax `d`, `round-half-away(x/d)`, crumb `n` of byte `m` covers element
  `m + n*32` of each 128-group.
- `quantizeRowTQ2_0ScaledInto` + `ternaryAbsmeanScale`: the BitNet b1.58
  recipe — per-tensor `d = clamp(mean|W|, 1e-5, ∞)`, `clamp(round(W/d),−1,+1)`
  (arXiv:2504.12285; the 1-bit-LLMs training FAQ). Every block stores the
  same `d`, so the result is plain valid TQ2_0.
- `quantizeRowForDType(.tq2_0, ...)` routes; `gguf.encodeF32`/`decodeF32`
  gained `.tq2_0` arms; `export-gguf --dtype/--experts-dtype tq2_0` works,
  with non-256-divisible tensors falling back to their source dtype exactly
  like the other 256-block targets.

## STE training op (training pipeline)

`Tensor.dotTernarySte(ctx, weight, contract_tag)` (`src/ag/tensor.zig`):
latent f32 weight `[out, in]`, forward = absmean-encode + mul-free f32
matmul; backward = `dx = gy · dequant(W_q)` (the quantized weights, matching
what the forward computed) and `dW = gyᵀ·x` — the pure identity STE, no
clipping or masking, exactly the BitNet recipe (`w + (Q(w) − w).detach()`).
The encoded blocks live in the backward node and are freed with it. Pinned
by gradcheck (dx), an explicit STE-identity test (dW ≡ plain matmul VJP),
and exec-scope lifecycle tests.

## Ternary-native evolution strategies (training = inference)

`src/es.zig` grows a second slot kind: genomes ARE packed `[]BlockTQ2_0`
(the block scales `d` are never touched — fix them at init, e.g.
`1/sqrt(k·2/3)` for uniform trits). No latent floats exist for ternary
slots, so the state you train is byte-for-byte the state you serve, and
members are evaluated through the real int8 flagship kernels.

Adapted from EGGROLL's integer recipe (arXiv:2511.16652, App. H — their
finding that sparse single-bin updates *improve* stability transfers whole):

- **perturb**: sparse trit flips regenerated from a counter stream in a new
  `es_trits` RNG domain (a pure function of seed/iteration/member — the
  existing O(1)-memory contract); `max(1, rate·len)` flips of ±1 with clamp;
  antithetic odd members mirror deltas. Restore replays a sparse
  (index, old-crumb) undo log in reverse — clamping is lossy, so
  regenerate-subtract cannot work for ternary.
- **update**: the existing reward shaping (z-score / centered-ranks /
  antithetic fold) feeds fitness-weighted votes on touched indices; the
  top-K by |vote| (ties by index — deterministic) move **one bin** toward
  sign(vote), clamped; `K = round(update_fraction·len / (1 + decay·t))`.
- Config: `ternary_flip_rate` (0.001), `ternary_update_fraction` (0.005),
  `ternary_update_decay` (0.0) — checkpoint contracts, persisted as
  `es_ternary_*` in `TrainerState`. Float and ternary slots coexist in one
  trainer (biases/scales stay Gaussian-ES floats); the float noise streams
  are untouched (pinned bitwise by a mixed-trainer test).

`zig build es-ternary-spirals` is the acceptance demo: a 2→256→256→2 MLP
whose hidden and output layers are packed ternary genomes trained from
random trits by ES — every member evaluation runs `quantizeRowsQ8_K` +
`matmulTQ2_0RhsRange`, i.e. the deployed inference path, and the run
self-verifies.

## Limits and future work

- k (contract dim) must be a multiple of 256 everywhere (block granularity);
  `dotTernarySte` rejects other shapes with a clear error.
- `tq1_0` remains decode/cold-matmul only.
- TL2-style LUT kernels (1.67 bpw, pshufb/tbl) — worth revisiting only for
  non-dot-product CPUs or footprint-bound deployments.
- `dotTernarySte` re-encodes the latent weight every forward (inherent to
  STE); a pre-encoded-RHS overload for frozen/tied weights is future work.
- The ES vote sort is a full sort of touched entries; a partial top-K select
  under the same pinned order is a drop-in future optimization, and the
  flip/undo/vote engine could be factored over an element codec if a second
  discrete genome kind (int8 EGGROLL, TQ1_0) ever lands.
- GPU legs (Metal/CUDA) deliberately out of scope for this record.

## Provenance

Kernel-structure and format lineage: ggml/llama.cpp (MIT) — see
`docs/THIRD-PARTY-NOTICES.md`; local reference clone `refs/llama.cpp`
(`ggml/src/ggml-quants.c`, `ggml/src/ggml-cpu/arch/{arm,x86}/quants.c`).
Recipes: BitNet b1.58 (arXiv:2504.12285 + the 1-bit-LLMs training FAQ),
bitnet.cpp kernel taxonomy (arXiv:2410.16144, 2502.11880), T-MAC LUT
technique (arXiv:2407.00088) — compared against, not ported. ES update:
EGGROLL (arXiv:2511.16652), machinery adapted, no code ported.
