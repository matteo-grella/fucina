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

The new dense-f32 contract is:

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

CUDA uses eight reusable in-flight slots. Each slot owns grow-only A/B/C device
buffers and reusable input-ready/completion events. Ordinary f32 storage is
page-locked once with `cuMemHostRegister`; its `Resource` unregisters only when
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

F16 GEMM and the quantized grouped staging protocol retain their existing
blocking staging contract in this change. They share the persistent provider
resources but need typed/panel-specific ownership before they can use the same
`Work` seam. Dense f32 GEMM/GEMV and every dense tagged `dot` lowered to it are
the implemented async surface.

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

# on matteo@192.168.1.24
zig build test-fucina -Dgpu=cuda
OPENBLAS_NUM_THREADS=32 \
LIBRARY_PATH=/home/matteo/tools/openblas/lib \
LD_LIBRARY_PATH=/home/matteo/tools/openblas/lib \
zig build bench-gpu-dispatch -Dgpu=cuda -Dblas=openblas -Doptimize=ReleaseFast -- --iters 15 --queue 4
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

Provider tests add an edge-tile, two-GEMM dependency chain that performs no
host read between commands and compares the final result with an f64 reference.
The normal Metal and remote CUDA test roots pass, as do `cuda-check`, the
default core root, and `arch-check` (zero SCCs).
