# Memory Model — Buffer Pool vs Arena

This document records *why* Fucina manages transient tensor memory with a
per-tensor `create` → `defer x.deinit()` pattern backed by a reusable
`BufferPool`, and why an arena allocator was considered and **rejected**. It is
grounded in the source; file:line references (verified against the tree
2026-07-17) are included so the rationale can be re-verified rather than
trusted.

The short version: **keep the current pattern.** The `BufferPool` already *is*
an arena (it amortizes allocation), but a strictly better one — it additionally
recycles buffers *within* a single forward pass, bounds steady-state memory, and
coexists with refcounted views and the autograd tape. A generic
`std.heap.ArenaAllocator` (or per-token / per-forward reset arena) would regress
on all of those.

---

## 1. How the current pattern works

The dominant idiom in the LLM forward path is:

```zig
var x = try someOp(ctx, ...);
defer x.deinit();
```

That `deinit` is not a naive free — it is the *driver* of buffer recycling. The
chain is:

```
tensor.deinit()  →  buffer.release()  →  refcount hits 0  →  reclaim()  →  buffer returns to the free-list
   ag/tensor.zig:977   tensor.zig:177      storage.zig:120     exec/buffer_pool.zig:171
```

`ExecContext` (via its embedded `Runtime`) owns one `BufferPool`
(`src/exec/runtime.zig:49`; type at `src/exec/buffer_pool.zig:47`). It is:

- **A size-bucketed free-list.** `acquire(len)` (`src/exec/buffer_pool.zig:82`) does
  first-fit over a list kept **sorted ascending by `data.len`**, returning the
  first buffer with `data.len >= allocationLen(len)`; on a miss it creates a new
  `Buffer` with `reclaim` as the release callback. `reclaim` inserts *before*
  existing same-length entries, so within a size class reuse is LIFO — the
  most recently released buffer is handed back first.
- **Size-rounded.** `allocationLen` (`src/exec/buffer_pool.zig:273`): `len <= 1024` →
  `ceilPowerOfTwo`; else `alignForward(len, 1024)`. This collapses nearby
  logical sizes into shared buckets, which *helps* reuse.
- **Bounded.** `max_cached_bytes` caps the CACHED (free-list) bytes at 1 GiB
  (`src/exec/buffer_pool.zig:59` — raised from the original 64 MB so big prefill
  transients stay cached; per the code comment at `:54-58`, retention is bounded
  by the actual peak transient set, not by this cap). In `reclaim`
  (`src/exec/buffer_pool.zig:171`) a returned buffer is destroyed instead of
  cached if it alone exceeds the cap, or if adding it would; otherwise it is
  inserted (sorted) and `cached_bytes` is bumped.
- **Mutex-guarded**, with an atomic `outstanding` counter incremented on every
  `acquire` and decremented in `reclaim`; `BufferPool.deinit` asserts
  `outstanding == 0` (`src/exec/buffer_pool.zig:70`), i.e. no live pooled buffer
  may leak past context teardown.

The recycle invariant is asserted by a dedicated unit test: after `first.deinit()`,
the next same-size op returns `second.buffer == first_buffer`
(`src/exec_tests.zig:338-358`).

### What is and isn't pooled

The pool has **two arms sharing one byte budget** (`cached_bytes` /
`max_cached_bytes`):

- **The f32 arm** — a free list of `*storage.Buffer`. `ctx.empty` / `emptyRank`
  acquire from it (`src/exec/runtime.zig:179/:186`). In an LLM forward
  essentially all transient activations are f32 (every matmul/linear/norm/add
  output is a default-dtype `FloatTensor`), so this arm covers the hot path.
- **The byte-slab arm** — a free list of 64-byte-aligned, 4096-byte-rounded raw
  slabs (`[]align(64) u8`). `emptyTyped` / `emptyRankTyped` route every
  non-f32 dtype through `acquireTyped` (`src/exec/runtime.zig:193-207`), which
  wraps a slab in a typed `storage.BufferOf(dtype)` header whose release hook
  returns the slab to the free list (cross-dtype reuse: an f16 LHS-cast slab
  can serve q8_k scratch next op). Hot consumers inherited pooling with no
  call-site changes: the per-projection f16 LHS cast in
  `matmulTransB2DWithF16Rhs` (`src/exec/matmul.zig`), typed gathers, typed
  matmul outputs. Non-DType packed block scratch (the quantized-LHS layouts
  in the fused K-quant FFN paths, `src/exec/quant_matmul.zig`) uses the same
  arm via `acquireScratch(T, len)` leases.

What still allocates directly: the backend-tier LHS-quantization scratch below
the exec seam (`matmul2DQuantizedRhs*` in `src/backend/native.zig` — the
deliberate allocator exception noted in `AGENTS.md`; decode is covered by stack
fast paths, and pooling the prefill arm is a known deferred optimization), and
load-time RHS weight packs. The once-hot f16 temporary during KV-cache append
was fused away before pooling existed: rows cast straight into the f16 cache
slot (`src/llm/kv_cache.zig:269-280`).

---

## 2. Inference vs training: the two lifetime regimes

- **Inference tensors are constants** (`grad_state == null`, built via
  `fromTensor` / `fromSlice`; `src/ag/tensor.zig:244`). `deinit`
  (`src/ag/tensor.zig:977`) releases the raw buffer immediately, so it returns
  to the pool mid-pass. This is what makes the pool behave arena-like *for free*
  in inference.
- **Training variables retain their inputs.** Backward functions store operand
  values via `cloneView()` at op-execution time (`src/ag/backward.zig`: mul/div
  `:152-153`, relu `:480`, dot `DotBackward` `:5371` delegating to
  `EinsumBackward`, cloneViews at `:5422-5426`), and `cloneView` bumps the
  refcount (`src/tensor.zig:190`).
  Those input buffers therefore **cannot** return to the pool until the tape
  node is destroyed in/after `backward`.

"Wasting" memory on a buffer that will be read again is not waste — that is
exactly the training case, and it is inherent to needing activations for
backward, not something an arena would change.

---

## 3. Views are refcounted aliases (the decisive constraint)

Every view operation retains the source buffer and releases it on `deinit`:
`cloneView` (`src/tensor.zig:190`), `viewWithStrides(Offset)`
(`src/tensor.zig:196/:200`), `reshape` (`src/tensor.zig:224`), `broadcastTo`
(`src/tensor.zig:238`); `narrow` goes through `viewWithStridesOffset`
(`src/exec/gather_scatter.zig:80`). A view's lifetime is independent of its parent's.

The most important instance: per-step attention reads the KV cache via a
**zero-copy `narrow`** (`src/llm/gemma/gemma4.zig:993`) that aliases a
**session-lifetime, non-pooled f16 buffer** (`KvCache.k/v`, allocated once via
`emptyRankTyped(.f16, ...)`, `src/llm/kv_cache.zig:179`; `reset()` only sets
`len = 0` and keeps the buffers, `:199-200`). A region-reset arena has no per-object
lifetime to represent this — it would either free live KV memory (corruption) or
have to carve KV out entirely (at which point it is no longer one arena).

---

## 4. Why an arena was rejected

A `std.heap.ArenaAllocator` / per-token / per-forward reset fails on four
substantive axes (in addition to the view/KV constraint in §3):

1. **Peak memory regresses from working-set to sum-of-all-intermediates.** The
   pool reclaims a buffer the instant a transient dies; per-block peak live set
   is only ~6–12 tensors (`attnBlock`/`ffnBlock`, `src/llm/gemma/gemma4.zig:1031/:1126`),
   and the residual stream is a single carried `x` advanced via `ctx.replace`
   (which frees the old buffer each layer, `src/exec.zig:362`). An arena frees
   nothing until reset, so a forward balloons to roughly `n_layer ×` the
   activation footprint — strictly worse than the pool, whose steady-state
   retention is bounded by the actual peak transient set
   (`src/exec/buffer_pool.zig:54-59`).
2. **It destroys cache locality.** Bump allocation returns a fresh address per
   op; the pool returns the *same* address for same-sized successive
   allocations, keeping the hot working buffer warm in L1/L2. Address reuse is
   the asserted behavior (`src/exec_tests.zig:338-358`) and directly serves the
   "match/beat llama.cpp on CPU" North Star.
3. **It cannot express refcounted views / KV aliasing** — see §3.
4. **It is impossible for training** — activations must outlive the forward for
   backward (§2), so an intra-forward reset is incorrect by construction.

The pool already delivers the *only* real arena advantage — allocation
amortization — and adds intra-pass reuse plus a bounded cap the arena lacks.

### Alignment with project intent

- The pool is a **shipped, named feature** ("bucket-rounded buffer allocation
  for small temporaries", `README.md`), and the allocation contract built on it
  is listed under **Current Strengths**, not tech debt (`ARCHITECTURE.md`,
  "Current Strengths"). The per-tensor `defer x.deinit()` recycle idiom is the
  documented ownership model.
- House rules reinforce it: "Backend outputs are exec-supplied" and
  "Explicit ownership … slices/tensor views *borrow*; pair every allocation with
  deterministic `errdefer`/`defer`" (`AGENTS.md`, House rules).
- An arena is **never mentioned** in the docs. The only documented future memory
  direction is the *opposite* of a generic scope-arena: a model-session with
  statically **preallocated, semantically-bound** buffers, explicitly "not a
  generic ggml-like graph" (`README.md`, closing scope note), and gated by the
  "Eager and local" house rule — no fusion/compiler layer without a concrete
  design (`AGENTS.md`, House rules).

---

## 5. The one honest caveat: ergonomics

The genuine cost of the current pattern is **boilerplate** — ~21 `defer .deinit()`
lines across `attnBlock` + `ffnBlock` (`src/llm/gemma/gemma4.zig:1031/:1126`).
(The hand-written `catch { …deinit(); return e; }` error-path cleanups this
section originally cited have since been converted to plain `defer`/`errdefer`
arms.) These manual frees are the real fragility, and they are precisely the
cases an arena could not safely automate either.

If that boilerplate ever becomes a maintenance problem, the **only** sound
mitigation is a thin scoped **"frame" helper**: register transient tensors and
bulk-call `.deinit()` at block exit. Hard constraints:

- It must **release into the pool** (refcount `release`), **never** bump-free a
  memory region — otherwise it dangles every outstanding view and the KV alias.
- It must **exclude** the residual `x` (carried via `ctx.replace`), the KV
  cache, and any `requiresGrad()` tensor.
- It is a layer *on top of* the pool, not a replacement for the allocator, and
  per the "benchmark before done" house rule (`AGENTS.md`) it should be proven
  perf-neutral (e.g. via `zig build bench-facade`) before landing.

Recommendation: **don't add it unless the boilerplate actually bites.** The
pattern is mechanical and uniform, and `DebugAllocator` catches any slipped
`deinit` in tests.

**Adjudicated 2026-06-10:** generalizing the autograd exec scopes
(`ExecContext.openExecScope` — implicit ownership of training intermediates)
into a deinit-eliminating arena for the inference engines was evaluated and
**rejected**. The scope's release path is mechanically identical to a `defer
deinit` (both end in `BufferPool.reclaim`), but the *timing* inverts the
discipline this document defends: (i) a held scope turns the pool's O(1)
working set into O(N) live intermediates with cold addresses — pinned at
<=2 vs 16 outstanding buffers on a 16-op chain by the "exec scope holds
buffers until close" test in `src/ag/tensor_tests.zig`; (ii) scope adoption
covers only the f32-facade op tails, never the typed/quantized/raw ops the
engines run on, so a scoped engine still manages those explicitly. (A third
fact at adjudication time — `ctx.replace` double-freeing scope-owned results
— was since neutralized by the `scope_owned` flag, which makes `deinit` a
safe no-op on scope-owned borrows, arena-allocator style. That flag is what
makes forward code WRITE-ONCE: the engines' defer-deinit forwards can be
trained by opening a scope around them — without changing what unscoped
inference does.) Scopes remain correct where holding the graph is required
semantics (training) and harmless on cold no-grad paths; for pure inference,
deinit-ASAP with no scope stays the discipline, and the standing gate for any
inference frame helper is unchanged.

---

## 6. Known sharp edges (not arena arguments)

- **First-fit over an ascending free-list** takes the smallest cached buffer
  that fits, but `allocationLen` over-rounds, so an oversized buffer can be
  handed back (internal fragmentation up to the next 1024-aligned size; up to
  the next 4096-byte quantum on the slab arm). A pool tuning concern only.
- **Pool accounting now covers both arms** (fixed with the byte-slab arm:
  `cached_bytes` counts true bytes of cached f32 buffers *and* cached slabs
  under the one shared `max_cached_bytes` budget). What it still cannot see:
  allocations that never enter the pool — backend-tier LHS-quant scratch,
  load-time weight packs, tensors built via `fromSlice`-style constructors.
- **Session-lifetime typed buffers are pool-retained at teardown.** KV-cache
  f16 layers and resident-bf16 weights are `emptyTyped`-backed, so when a
  model/session is torn down (before the owning `ExecContext` dies) their
  slabs return to the free list — retained up to the cap instead of freed.
  This is deliberate (a feature for multi-session processes: the next session
  reuses warm slabs); a process that tears down a large session and keeps the
  context alive holds up to `max_cached_bytes` of cache. `ExecContext.deinit`
  frees everything.
- **`acquire`/`acquireSlab` release the mutex before allocating** a fresh
  buffer/slab on the miss path (`src/exec/buffer_pool.zig:82-108/:205-225`). Correct
  today (the new buffer is not yet shared, `outstanding` is atomic), but any
  future change touching shared pool state in that window must re-take the lock.
- **Typed pooled buffers must never be marked stable-lifetime GPU RHS.** The
  slab arm makes address reuse routine, so a cached GPU wrap keyed on a pooled
  transient's address would read stale data (`RhsLifetime` in
  `src/exec/quant_matmul.zig`; today every `.stable_process` caller wraps
  device-resident or mmap'd weight bytes, never pooled storage — keep it that
  way).
- **`KvCache.reset()` does not zero or free** — it only sets `len = 0`
  (`src/llm/kv_cache.zig:199-200`); stale f16 data is simply overwritten on the
  next append, which is correct only because attention reads just the `[0..len]`
  prefix.
