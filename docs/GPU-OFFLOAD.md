# Eager GPU Offload Without A Compute Graph

Fucina treats Metal/CUDA as callable accelerators inside the native backend.
An eligible dense GEMM/GEMV (including a tagged `dot` lowered to GEMM) is
validated and submitted when the eager op is called; no operation is recorded
for later planning, fusion, or replay. This document describes the completion
and storage protocol that removes the old per-call host stall while preserving
that model.

## The contract

The old provider contract was synchronous: encode/launch, wait for the whole
command/stream, copy back, return. It was simple, but it charged every op for a
host round trip even when the next action was another GPU GEMM or independent
CPU work.

The dense f32/f16 and stable-weight quantized contract (Q4_K/Q6_K/Q8_0 on
Metal; those plus Q5_K on CUDA) is:

1. validate and allocate the ordinary CPU-visible output through
   `ExecContext`;
2. encode and submit the GPU command immediately;
3. attach a type-erased `accelerator.Work` completion token to the output
   storage and return the ordinary tensor;
4. preserve device/queue order when a later GPU op consumes that tensor;
5. synchronize only when host visibility is actually required.

The token is completion metadata, not a compute node. It has no operation
description, shapes, function to replay, dependency scheduler, or optimizer.
The GPU command already exists before the tensor is returned. Consequently the
runtime remains eager and graphless.

Host visibility is forced by `Tensor.data*`, `item`, copies/materialization,
and every CPU kernel data accessor. Releasing the last storage reference also
waits before its memory/device slot can be recycled, but discards the result
without an unnecessary host copy. Views share the storage token, so a read
through any alias establishes readiness for all aliases.

## Persistent provider lanes

Both providers already created their expensive process resources lazily and
kept them open. The implementation makes that policy explicit:

- Metal keeps one `MTLDevice`, compiled library/pipeline cache, and
  `MTLCommandQueue`. Each eager op uses a one-shot command buffer, commits it
  immediately, and returns a retained ticket. Command buffers cannot be reused
  after commit; keeping one uncommitted across calls delayed the first op and
  prevented useful CPU/GPU overlap, so it was rejected. Queue order provides
  dependency order across the one-shot buffers.
- CUDA keeps the primary context, dedicated upload/compute/download streams,
  and cuBLAS handle open. cuBLAS stays bound to the compute stream; reusable
  events connect the three lanes without a device-wide synchronization.

Immediate commit is important: an “open” command buffer that waits for an
unknown future op is batching by latency inflation. Fucina instead keeps the
queue/streams persistent, commits every eager call, and batches only the host
waits.

## Storage and transfer reuse

`src/accelerator.zig` defines two backend-neutral lifetime objects:

- `Work`: refcounted submitted-work token with `ensureHost`, discard, and an
  optional device-result address;
- `Resource`: a non-compute cache entry owned by one backing storage
  allocation.

`storage.BufferOf` owns one pending writer token, one latest-reader token, and
one resource. Writer work is consumed on a CPU read or final release; mutable
CPU access additionally waits for the latest device reader. A resource
survives ordinary `BufferPool` release/reuse and is destroyed only when the
backing allocation is destroyed.

Metal uses `Resource` for one persistent `MTLBuffer` page wrapper per storage
allocation. Pool reuse changes values but not the allocation mapping, so this
is safe and removes repeated Objective-C allocation/VM wiring. The async ticket
retains its command and all referenced Metal buffers; Zig also retains both
input storage buffers until completion because the command uses unretained
resource references for lower submission overhead.

CUDA uses eight reusable in-flight slots. Each slot owns grow-only typed A/B/C
and auxiliary tile buffers, a grow-only split-K partial buffer, plus reusable
input-ready/completion events.
Ordinary f32/f16 storage is page-locked once with `cuMemHostRegister`; its `Resource` unregisters only when
the backing allocation is destroyed, so `BufferPool` reuse amortizes page
registration. H2D runs on the upload stream, which records an event consumed by
the compute stream. A resident RHS is used by device address with no transfer.
At a CPU boundary the download stream waits on the compute event and copies
directly into the registered exec-supplied tensor—there is no staging-to-tensor
memcpy and the calling CPU sees only the final stream fence. Unsupported host
allocations retain a correct pinned-stage fallback. If a dependent GPU op
consumes a pending output, it retains the producer token and uses the
producer's C device buffer directly: there is no D2H/H2D bounce between calls.

Eight slots bound transient device-buffer growth. Host page-locking is bounded
by the existing `BufferPool` retention budget and released with each backing
allocation. If all slots are live, GPU submission declines and native dispatch
safely falls through to CPU; it never allocates an unbounded hidden queue.

## Ordering and unavoidable synchronization

The public ordering remains program order:

- GPU → GPU on the same provider: persistent queue/stream event order;
  dependent CUDA calls also retain the producer slot/device pointer.
- GPU → CPU: the first CPU data access waits and makes the host bytes visible.
- CPU → GPU: ordinary host inputs are staged before the kernel; resident bytes
  are read in place/by managed device address.
- GPU reader → CPU mutation: mutable `data*` access waits for the latest
  command using that storage. Const reads may overlap a read-only device use.
- final `deinit`: waits for device use before returning pooled storage or a
  device slot, but skips D2H when nobody reads the result.
- context/model teardown: existing ownership order releases tensors before
  their pools/resident storage.

The runtime can fence mutations made through a tensor handle; it cannot
observe writes made directly through the caller-owned slice behind
`fromBorrowedSlice`. Before changing such a slice while an eager GPU call may
still be using it, call the tensor's mutable `data()` accessor (which performs
the reader fence) or synchronize externally. Read-only borrowed weights do not
have this hazard.

A deferred device execution error is discovered at the host boundary and is
fatal. Replaying the original call on CPU would require retaining an operation
description and all operands after return—effectively a graph—so submission
failures fall back before return, while post-submit device faults do not replay.

F16 NT GEMM uses the same completion seam. Metal instantiates the steel kernel
with f16 inputs and a direct f32 output; CUDA asks `cublasGemmEx` for f32 C.
This removes the old process-global f16 staging lock, f16 result buffer, CPU
widen pass, and an unnecessary output rounding step.

Stable-weight dense quantized linears also bind the ordinary input/output
tensor storage directly. Metal copies at most 4 KiB of 32-row tile descriptors
into command-buffer-owned bytes and supports up to 8192 rows in one eager
submission. CUDA keeps a pinned/device tile pair in each in-flight slot and
launches the vendored dequant kernel on the persistent compute stream. On
compute capability 7 or newer, Q4_K/Q5_K/Q6_K/Q8_0 prefill uses f16-input WMMA with
f32 accumulation after dequantizing to the same half-rounded shared operands as
the original scalar-FFMA kernel. The unsplit/grouped launcher chooses a
32-column tile only when a 64-column grid would fill less than two thirds of
the SMs; ordinary LLM shapes use 64 columns to avoid duplicating activation
loads. Full tiles store WMMA fragments directly to the exec-owned output and
edge tiles use a guarded shared epilogue. F32 activation loads and dequantized
weight registers are
packed into shared half storage with vector loads and `half2` conversions.
When the N64 output grid fills less than roughly seven eighths of the SMs,
dense prefill partitions K two ways (up to three for Q6_K), writes reusable
partial planes, and queues a fixed-order reduction on the same compute stream
before recording completion.
There is no host fence or steady-state allocation. The scalar kernel remains
the compatibility and diagnostic fallback (`FUCINA_GPU_QUANT_MMA=0`), while
`FUCINA_GPU_QUANT_SPLIT_K=0` isolates the unsplit tensor-core path. A
shared-input batch encodes/launches one weight matrix after another without
replicating activation rows. Transient quantized RHS slices retain the blocking
path: deferring a command past the lifetime of an unowned byte borrow would be
incorrect.

### What the ik_llama.cpp audit contributed

The CUDA MMQ implementation in `ik_llama.cpp` was audited at commit
`b90939934add9ba4fbb37e8c6470809a70b78f0a` (MIT), principally
`ggml/src/ggml-cuda/mmq.cuh`, `mma.cuh`, and `quantize.cu`.

Its synchronization machinery confirmed that Fucina was not missing another
simple eager-provider trick. Ordinary ik_llama execution also retains
nonblocking streams, reusable events, and pooled allocations. Its remaining
launch-amortization mechanism is CUDA Graph capture of a complete ggml graph;
that requires the operation sequence Fucina deliberately does not own and is
therefore not portable to this eager API.

The compatible idea was MMQ's Stream-K work partitioning. Fucina uses a
smaller split-K specialization rather than copying that scheduler: only an
underfilled dense eager WMMA launch is split, the normal output tiling remains
unchanged, partial storage belongs to the existing bounded slot, and the
reduction is another immediately submitted command on the same stream. Grouped
MoE retains its phase scheduler and does not use this path. This preserves the
callable-accelerator model while filling otherwise-idle SMs.

The audit also identified a larger possible next step, not folded into this
change. ik_llama quantizes each f32 activation tile once to a Q8_1-style int8
layout (including scales/sums), unpacks quantized weights into a signed-int8
shared layout, and uses int8 MMA. That can reduce repeated dequant/half-convert
work, but it changes activation numerics and requires architecture-specific
PTX variants: Fucina's portable committed module currently targets
`compute_70`, while the useful signed-int8 MMA instructions differ across
Turing and Ampere/Ada. It should be treated as a separately parity-gated
backend rather than hidden inside the current f16-WMMA numerical contract.

Grouped MoE remains phase-synchronous by necessity: CPU gather feeds gate/up,
CPU GeGLU consumes that output and feeds down, and CPU scatter consumes down.

The Q5_K grouped kernel is implemented and parity-tested at the provider's
tile-table API, but no current model-specific MoE loader routes Q5_K expert
stacks to it. Gemma/Diffusion currently use Q4_K or Q6_K gate/up plus Q8_0
down; other Q5_K expert layouts retain their CPU orchestration. Dense Q5_K
linears are the production model path added here.
Metal waits at those two CPU boundaries over unified memory. CUDA now queues
panel/tile H2D, kernel, and panel D2H through its persistent streams/events and
performs one host fence at each boundary; it no longer synchronizes compute
before starting the download. These are data-dependency fences, not
per-dispatch setup overhead.

## Gates

The large-GEMM gate remains work- and shape-based. On M1 Max the conservative
cold single-op floor is now `2^32`: isolated 1024³ and 2048×1024×1024 trials
were DVFS-sensitive crossovers, while 2048×2048×1024 won consistently. CUDA
uses `2^27` (`FUCINA_GPU_MIN_WORK_RESIDENT`) for an already device-resident
RHS and retains the `2^33` transient-RHS PCIe floor. A resident-GEMV gate
allows `m <= 8`, `n,k >= 256`, and `m*n*k >= 2^24` only when the RHS already
has a persistent mapping/device address; `FUCINA_GPU_MIN_WORK_GEMV` overrides
that floor. CUDA refuses a nonresident decode RHS; Metal accepts a resident or
already storage-mapped RHS because unified memory needs no PCIe weight copy.

Metal also admits SMALL-m 16-bit-weight GEMMs (f16/bf16 RHS) whose weights
already carry a storage-lifetime page wrap, floored at `2^27`
(`FUCINA_GPU_MIN_WORK_16BIT_RESIDENT`). The floor is measured, not
theoretical: admitting m=4 batched-decode projections at `2^24` LOST 18%
end-to-end on an M1 Max (per-dispatch overhead outweighs the whole-matrix
bandwidth win at that width, unlike CUDA's resident admission), while m>=16
admission is neutral-to-positive.

Related CPU-side knob (non-GPU builds only): `FUCINA_CPU_F32_SHADOW=1`
routes prefill-shaped 16-bit-weight GEMMs (`m >= 32`,
`FUCINA_CPU_F32_SHADOW_MIN_M`) through the BLAS f32 arm over a widen-once
f32 shadow cached on the weight's storage (+4 bytes/weight resident;
weights must not be trained in place). Measured on Qwen3-1.7B-BF16
self-study (M1 Max + Accelerate): 2.2x end-to-end (28.5 -> 12.7
s/conversation) with identical per-step losses; at the GEMM level BLAS wins
1.5-2.5x for m >= 32 while decode stays with the 16-bit streaming kernels
(half the bytes per weight).

F16 uses `2^27` on Metal and for streamed CUDA prefill. CUDA has a separate
resident f16 floor (`2^20`): a 1×4096×1024 resident call measured 18.3 µs on
the RTX host versus 77.4 µs on CPU, while nonresident decode is still refused
by the m≥32 transient gate. Dense quantized gates distinguish the CPU
competitor. Compact/raw fallbacks retain the Parakeet/MoE thresholds; GGUF
model weights that already own a load-time-packed CPU fallback use measured
per-format floors. Metal defaults are Q4_K `2^30`, Q6_K `2^31`, Q8_0 `2^29`;
CUDA defaults are Q4_K `2^27`, Q6_K/Q8_0 `2^24`.

The explicit equal-shape vector `dot` returns one scalar and stays on CPU: a
GPU launch plus a mandatory one-value host fence cannot amortize. Tagged
contractions that lower to dense matmul inherit the GEMM/GEMV gates.

The measurements below show why the general small-op gate must remain. Async
submission removes the caller stall; it does not make a 256³ GPU kernel faster
than Apple AMX.

## Verification and measurements

Commands (ReleaseFast for measurements):

```sh
zig build test-fucina -Dgpu=metal
zig build bench-gpu-dispatch -Dgpu=metal -Doptimize=ReleaseFast -- --iters 15 --queue 4
zig build bench-gpu-dispatch -Dgpu=metal -Doptimize=ReleaseFast -- --shape 'gemm 1024^3' --iters 63 --crossover
zig build bench-gpu-formats -Dgpu=metal -Doptimize=ReleaseFast -- --iters 5 --queue 4

# on matteo@192.168.1.24
zig build test-fucina -Dgpu=cuda
OPENBLAS_NUM_THREADS=32 \
LIBRARY_PATH=/home/matteo/tools/openblas/lib \
LD_LIBRARY_PATH=/home/matteo/tools/openblas/lib \
zig build bench-gpu-dispatch -Dgpu=cuda -Dblas=openblas -Doptimize=ReleaseFast -- --iters 15 --queue 4
zig build bench-gpu-formats -Dgpu=cuda -Dmax-threads=32 -Doptimize=ReleaseFast -- --workers 31 --iters 5 --queue 4
```

`bench-gpu-dispatch` uses a resident RHS, reports median wall time, and checks
the async result against CPU CBLAS (`max_abs <= 5e-3`). “sync” is the former
blocking provider entry; “async” includes submit plus the eventual host read;
“submit” stops before the host fence; queue throughput submits four independent
eager calls and then reads them.

### Metal — Apple M1 Max, Accelerate/AMX

| Shape | CPU BLAS µs | old sync µs | async host-visible µs | submit µs | queue-4 GF/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMV 1×4096×4096 | 1895.3 | 1121.3 | 1221.8 | 10.0 | 46.0 |
| GEMM 256³ | 33.3 | 275.5 | 233.4 | 6.2 | 411.9 |
| GEMM 512³ | 148.5 | 436.1 | 345.2 | 7.0 | 1378.2 |
| GEMM 1024³ | 1149.3 | 1521.1 | 1206.9 | 9.2 | 2070.1 |
| GEMM 2048×1024×1024 | 2189.5 | 2059.7 | 1801.8 | 14.0 | 4070.6 |
| GEMM 2048³ | 10030.2 | 5535.7 | 3604.8 | 22.5 | 5036.2 |

Those interleaved numbers show the benefit once the GPU is warm, but a gate
must also protect an isolated cold eager call. Five separate 63-sample 1024³
trials alternated CPU-first/GPU-first order: Metal's median ranged from 6%
faster to 25% slower and lost three trials decisively. In a traced trial the
Metal kernel averaged 0.953 ms—equal to Accelerate's 0.953 ms median—but
submission/scheduling/wait made host-visible Metal 1.216 ms. The 2^31-work
2048×1024×1024 shape was likewise a tie across three trials. At 2^32 work
(2048×2048×1024), Metal won all three trials by at least 24%; that is the cold
default. Resident GEMV still wins 1.55× in the interleaved run and 2048³ wins
2.78×. `FUCINA_GPU_MIN_WORK` remains available for a sustained GPU-heavy
workload whose preferred threshold is lower.

A final skeptical re-audit on the completed tree confirmed the conservative
choice. Two independent 31-pair 1024³ runs had Accelerate ahead by 14% and
20% (CPU/GPU medians 1089.5/1241.3 µs and 1060.4/1270.5 µs). At 2^31 work,
Accelerate still led 1996.9/2129.6 µs. At the configured 2^32 floor Metal led
5491.0/4056.2 µs, a 1.35× win. In other words, the default does not claim a
Metal win at the disputed 1024³ size.

### CUDA — RTX 5000 Ada Laptop, driver 580.126.18

Host: Intel i9-13950HX. CPU comparison is the host's custom OpenBLAS 0.3.29
Haswell build. A thread sweep (8/16/24/32) made 32 the fastest setting on this
hybrid CPU; `LIBRARY_PATH`/`LD_LIBRARY_PATH` point Zig and the runtime at its
non-system install.

| Shape | OpenBLAS-32 µs | old sync µs | async host-visible µs | submit µs | queue-4 GF/s | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GEMV 1×4096×4096 | 1059.9 | 87.7 | 88.3 | 8.1 | 558.8 | 1.04e-6 |
| GEMM 256³ | 57.6 | 93.4 | 66.6 | 7.0 | 815.2 | 1.49e-7 |
| GEMM 512³ | 256.4 | 314.6 | 225.1 | 7.7 | 1632.4 | 3.58e-7 |
| GEMM 1024³ | 2010.8 | 1005.1 | 881.8 | 16.3 | 3292.4 | 3.87e-7 |
| GEMM 2048×1024×1024 | 3833.7 | 1687.6 | 1536.4 | 14.1 | 3537.2 | 1.19e-6 |
| GEMM 2048³ | 14908.9 | 4061.2 | 3771.3 | 25.9 | 7137.4 | 1.73e-6 |

Against the actual OpenBLAS baseline, resident GPU async loses at 256³,
wins only 1.14× at 512³, then wins 2.28× at 1024³, 2.50× at
2048×1024×1024, and 3.95× at 2048³; the resident GEMV wins 12.0×. A focused
63-sample 512³ crossover was a narrow 1.05× GPU win, while 640³ and 768³ won
1.50× and 1.56×. The resident default starts at 512³ and the ordinary
nonresident floor remains far higher. Compared with the first pinned-stage implementation,
registered direct DMA plus the three-lane event chain cuts 2048³ host-visible
latency from 5.10 to 3.75 ms and raises queue-4 throughput from 3.43 to 6.50
TFLOP/s (7.14 TFLOP/s in the final OpenBLAS-paired run). Submission returns in
7–26 µs.

The final OpenBLAS-32 crossover audit measured 512³ at 230.9/224.1 µs
(only a 1.03× GPU edge), 640³ at 516.5/346.3 µs (1.49×), and 1024³ at
1777.4/879.9 µs (2.02×). This is the custom OpenBLAS 0.3.29 installation at
`/home/matteo/tools/openblas`, not the system `libblas` fallback.

### F16 and GGUF quantized linears

`bench-gpu-formats` compares the actual Fucina CPU competitors—its f16 row
kernel and load-time-packed quant kernels—with one eager GPU call over resident
GGUF weights. GPU time includes the eventual host visibility fence; submit
time stops immediately after commit/launch. Results are checked pairwise
(observed maximum absolute differences below 0.009 for quant and below 0.002
for Metal f16; CUDA f16 was below 3e-6). The f16 rows are a kernel-level paired
comparison: Fucina's public f32-activation/f16-weight linear first rounds its
LHS into a pooled f16 temporary on CPU for either contender, so that common
conversion is excluded. Any pending producer is necessarily made host-visible
at that conversion boundary, consistent with the CPU-resident eager model.

Representative Metal results (M1 Max, 7 workers + caller):

| Format/shape | CPU µs | GPU host-visible µs | submit µs | queue-4 GF/s |
| --- | ---: | ---: | ---: | ---: |
| f16 32×4096×1024 | 757.5 | 302.3 | 9.5 | 2178.7 |
| f16 128×4096×1024 | 1929.0 | 987.6 | 14.3 | 1652.6 |
| f16 1×151936×1024 lm-head | 3129.6 | 2948.7 | 26.8 | 149.9 |
| Q4_K 32×4096×4096 | 925.0 | 1020.9 | 8.1 | 1484.1 |
| Q4_K 128×4096×4096 | 3354.3 | 1213.8 | 10.9 | 6254.4 |
| Q6_K 64×4096×4096 | 1498.4 | 1499.5 | 10.8 | 2756.1 |
| Q6_K 128×4096×4096 | 3532.8 | 1497.9 | 15.5 | 4695.7 |
| Q8_0 32×4096×4096 | 1168.5 | 902.5 | 9.7 | 1695.7 |
| Q8_0 128×4096×4096 | 2935.6 | 1305.3 | 13.1 | 5159.6 |

These are why Metal's packed-CPU gates differ by format: Q4_K crosses near
2^30 work, Q6_K is only at parity there and waits for 2^31, while Q8_0 already
wins at 2^29. A 32×4096×1024 Qwen-sized quantized projection remained CPU
favored for all three formats (202–263 µs CPU versus 382–425 µs Metal), so it
is deliberately below every Metal packed gate. Small quantized decode remains
CPU-only on Metal.

Representative CUDA results (RTX 5000 Ada Laptop, 31 workers + caller for the
packed CPU comparison):

| Format/shape | CPU µs | GPU host-visible µs | submit µs | queue-4 GF/s |
| --- | ---: | ---: | ---: | ---: |
| f16 1×4096×1024 resident | 77.4 | 18.3 | 3.8 | 697.7 |
| f16 32×4096×1024 resident | 725.3 | 87.1 | 3.9 | 4374.3 |
| f16 128×4096×1024 resident | 1774.8 | 237.5 | 3.9 | 5240.1 |
| f16 1×151936×1024 resident | 8232.8 | 797.6 | 3.9 | 421.7 |
| Q4_K 32×4096×1024 | 267.8 | 108.6 | 6.0 | 4074.7 |
| Q5_K 32×1024×512 | 63.0 | 46.4 | 5.7 | 1451.3 |
| Q5_K 32×4096×1024 | 263.7 | 134.8 | 6.3 | 3307.8 |
| Q6_K 32×4096×1024 | 881.1 | 111.5 | 5.9 | 4013.4 |
| Q8_0 32×4096×1024 | 653.2 | 111.0 | 6.3 | 4133.5 |
| Q4_K 128×4096×4096 | 4212.4 | 1035.1 | 4.0 | 5788.6 |
| Q5_K 128×4096×4096 | 5577.7 | 1409.8 | 16.9 | 3854.2 |
| Q6_K 128×4096×4096 | 8144.7 | 844.3 | 3.7 | 7648.5 |
| Q8_0 128×4096×4096 | 4038.9 | 973.6 | 4.0 | 6246.0 |

Relative to the pre-audit f16-WMMA implementation, vectorized shared
conversion plus split-K changed the most underfilled shapes as follows (same
RTX host, queue four):

| Format/shape | old → final GPU µs | latency | old → final queue GF/s | throughput |
| --- | ---: | ---: | ---: | ---: |
| Q4_K 32×1536×512 | 52.8 → 44.1 | -16.5% | 1709.2 → 1854.1 | +8.5% |
| Q4_K 32×4096×4096 | 340.4 → 283.7 | -16.7% | 4184.1 → 5254.8 | +25.6% |
| Q6_K 32×1536×512 | 52.4 → 44.8 | -14.5% | 1669.6 → 1834.0 | +9.8% |
| Q6_K 32×4096×4096 | 282.1 → 237.7 | -15.7% | 5316.0 → 6802.5 | +28.0% |
| Q8_0 32×1536×512 | 52.2 → 44.4 | -14.9% | 1798.9 → 1876.0 | +4.3% |
| Q8_0 32×4096×4096 | 308.1 → 269.9 | -12.4% | 4687.4 → 5772.9 | +23.2% |

All six final rows retained the same displayed CPU-reference maximum errors
as the unsplit implementation (1.96e-3 to 6.06e-3). A same-binary
split-on/off A/B/A/B of Qwen3-0.6B Q6_K pp32 averaged 749.7 versus 725.0 tok/s
(+3.4%); individual process summaries were noisy on the laptop. Qwen3-4B
Q4_K_M pp32 remained effectively flat (155.1 versus 154.1 tok/s), showing that
host attention and other CPU boundaries can absorb an op-level gain.

The committed PTX is generated through NVRTC by `tools/gen_cuda_ptx.zig`, the
same frontend as `FUCINA_GPU_KERNELS=src`. This is performance-significant on
the reference CUDA 12.0 toolkit: an interleaved Q6_K 32×4096×4096 audit put
NVCC-generated PTX at 251.4 µs / 6382.8 GF/s and NVRTC at 241.3 µs / 6691.3
GF/s. The regenerated committed artifact reproduced 241.1 µs / 6685.5 GF/s.
All three results had the same 6.06e-3 maximum CPU-reference error.

Against the scalar CUDA kernel in the same nine-iteration run, WMMA raised
queued throughput by 11–37% over every non-decode Q4_K/Q6_K/Q8_0 shape in the
suite. At the two representative Qwen/prefill-128 shapes the gains were
respectively 14%/22% (Q4_K), 11%/19% (Q6_K), and 22%/24% (Q8_0). Decode is
intentionally unchanged: it uses the separate GEMV kernel.

Nsight Systems on Q4_K 128×4096×4096 recorded a 4×64 = 256-block launch on
the 76-SM Ada GPU, 16×16 threads/block, 40 registers/thread, 14 KiB static
shared memory, and a 0.704 ms average kernel; `ptxas` reported no spills. Thus
every SM is fed; the remaining gap to peak tensor-core FLOP/s is the fused dequant/shared-load
work, not an idle-core geometry bug. The same trace measured each 2 MiB
activation/output PCIe copy at about 0.18 ms. Those unavoidable host boundaries
explain why the whole-runner effect is smaller: Qwen3-0.6B Q6_K at pp128 moved
from 1308 tok/s (mean of the scalar-kernel A/A legs) to 1332 tok/s (1.8%),
whereas pp841 was flat within variance because attention/CPU work dominated.
Qwen3-4B Q4_K_M pp128 was also effectively flat in an A/B/A run (197 vs 198
tok/s). The 11–37% claim is therefore deliberately an offloaded-op throughput
claim, not a claim that every end-to-end prompt gains that amount.

Q5_K uses a format-specific decode crossover because the CPU switches from
compact blocks to its lane-packed x8 kernel at row four. With 31 workers plus
the caller, 1×4096² stayed on CPU (95.8 versus 108.6 µs), while
1×6144×4096 won on CUDA (212.5 versus 159.4 µs); rows 2, 4, and 8 at 4096²
were 230.0→184.1, 362.9→287.7, and 528.8→298.4 µs. Q5_K therefore uses
GEMV for rows 1–3, tiled MMA for rows 4–8, and a default
`FUCINA_GPU_MIN_WORK_DECODE_Q5=3·2^23` gate. The global CUDA quant-decode arm
remains opt-in. The resident f16 result is different: even small decode wins decisively because
only the activation/output cross PCIe, hence its residency-aware gate.

Provider tests add an edge-tile, two-GEMM dependency chain that performs no
host read between commands, direct-f32 f16 checks, and shared-input async
Q4_K/Q5_K/Q6_K/Q8_0 checks against dequantized CPU references. They also mutate an input after
submission to prove the reader fence. CUDA's quant test forces both WMMA and
scalar kernels, checks each against the CPU reference, and compares them
directly. Q5_K additionally covers edge tiles, grouped expert tiles, direct
eager storage, split-K, and GEMV; the parity-checking benchmark covers the
m=4 tiled decode transition. Both NVRTC-source and committed-PTX test roots
pass. The grouped CUDA tests exercise the
persistent upload/compute/download event chain.
The normal Metal and remote CUDA test roots pass, as do `cuda-check`, the
default core root, and `arch-check` (zero SCCs).

Mixed-format `*_K_M` fusion candidates are initially loaded without device
copies. When fusion declines, `fuseLinear` now restores ordinary per-part GPU
residency; otherwise each eager prefill would stream the same weights on every
pass. On Qwen3-0.6B-Q5_K_M tracing changed from 84 streamed calls to zero and
reported 330 resident asynchronous quant submissions in the timed pp32 pass.

Q5_K model-level arithmetic is tolerance-equivalent, not bit-identical to the
CPU packed kernel: CPU first quantizes activations to Q8_K, while CUDA consumes
f16-rounded activations and dequantized weights. On Qwen3-0.6B-Q5_K_M a fixed
32-token prompt produced the exact same 32-token greedy continuation; warm
prefill improved 503.3→770.3 tok/s at 32 tokens and 620.3→1167.1 tok/s at 128,
while opt-in decode improved 62.85→92.30 tok/s. A Q5_K_S 38-token prose prompt
is the recorded counterexample to any stronger parity claim: max/rms final-logit
error was 0.980/0.226 and a 0.10 CPU top-two margin reversed. Direct op tests
remain the correctness oracle because they compare each CUDA result to an
explicitly dequantized f32 reference (representative maximum absolute errors
2.02e-3 to 7.39e-3).
