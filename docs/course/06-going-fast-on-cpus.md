# Chapter 06 — Going fast on CPUs

*Part II — The tensor core*

In [Chapter 5](05-the-operation-library.md) we built the operation library: `ExecContext` validates shapes once, allocates the output, and dispatches to a backend kernel. Everything above that dispatch line is bookkeeping. Everything below it is arithmetic on slices — and that arithmetic is where essentially all of a neural network's wall-clock time goes.

This is the close-to-metal chapter. By the end of it you will have seen how one tree of portable Zig source compiles to NEON on an M1 and AVX2 on an x86 laptop, how a matrix multiply is taken from three obvious loops to a cache-blocked, register-tiled kernel that is 5.6× faster at large sizes, how a team of worker threads splits kernels across cores without paying a syscall per operation, and — just as important — how every one of those fast paths is held accountable by a deliberately boring reference implementation and a written benchmark protocol that records losses next to wins.

The theme of the whole chapter is a single discipline: **make it obviously right first, then make it fast, and never let the fast version escape the slow version's judgment.**

## 6.1 Where the time goes

Run any transformer forward pass under a profiler and the picture is lopsided: matrix multiplication dominates, usually by an order of magnitude over everything else combined. A linear layer is a matmul. Attention scores are a matmul. The attention-weighted values are a matmul. The feed-forward block is two or three matmuls. The vocabulary projection at the end is one enormous matmul. The elementwise glue between them — activations, additions, normalizations — touches each value once or twice, while a matmul accumulates `k` products into every output element and re-reads each input value once per output it feeds.

> **ML note** — The universal naming convention: a matrix multiply `C = A·B` has `A` of shape `m×k`, `B` of shape `k×n`, and `C` of shape `m×n`. It performs `m·n·k` multiply-adds. In a transformer, `m` is typically the number of tokens being processed at once: large during *prefill* (reading the prompt), and 1 during *decode* (generating one token at a time). That single parameter changes the performance character of the entire computation, and you will see it drive dispatch decisions throughout this chapter. The acronym GEMM ("GEneral Matrix Multiply", from the BLAS libraries) is used interchangeably with matmul; at `m = 1` the GEMM degenerates to a GEMV (matrix-*vector* multiply), and the economics flip — every weight byte is read for a single multiply-add, so single-token decode is memory-bandwidth-bound, while prefill and training reuse each weight `m` times and are compute-bound.

So the recipe for a fast CPU runtime is unromantic:

1. Make matmul fast — SIMD, cache blocking, threading.
2. Make everything else not slow — vectorize the elementwise pass, avoid allocation, avoid synchronization.
3. Measure, honestly, on the machines you actually care about.

Fucina targets two instruction-set architectures in earnest — Apple Silicon (NEON) and modern x86-64 (AVX2, with AVX-VNNI where present) — and the interesting engineering constraint is that it does so *from one source tree*, specialized at compile time. There is no runtime CPU dispatch, no function-pointer tables keyed on `cpuid`, and no `#ifdef` forest. The language does the work.

## 6.2 Two backends, chosen at compile time

Fucina has exactly **two** CPU backends:

- **`scalar`** (`src/backend/cpu.zig`) — plain serial loops. No SIMD, no BLAS, no threads. It exists to be obviously correct.
- **`native`** (`src/backend/native.zig`) — the production backend: portable `@Vector` kernels, an optional BLAS tier for large f32 GEMM, an optional GPU offload seam, and a worker team.

(`-Dbackend=cpu` is accepted as a deprecated alias for `scalar` — it is not a third backend. BLAS and the GPU providers are tiers *inside* the native backend, not backends of their own.)

The selection is a comptime switch over a build option, and the entire mechanism fits in a few lines (from `src/backend.zig:104–122`):

```zig
pub const Kind = enum {
    scalar,
    native,
};

pub const active_kind: Kind = switch (build_options.backend_kind) {
    .scalar, .cpu => .scalar,
    .native => .native,
};

// …

const active = switch (build_options.backend_kind) {
    .scalar, .cpu => scalar_impl,
    .native => native_impl,
};
```

Every method on the public `Backend` struct forwards to `active.<fn>`. Since `active` is resolved at compile time, the losing module is dead code: it is parsed but never dispatched to, its symbols never make it into the binary, and there is no `if (use_simd)` branch anywhere in a hot loop.

> **Zig note** — This is Zig's *lazy analysis* at work. A `switch` on a comptime-known value is resolved during compilation, and Zig only semantically analyzes code that is actually referenced. The unselected backend module doesn't just get optimized out later — most of it is never even type-checked into the build. The same pattern selects the GPU provider in `src/backend/gpu.zig`: a comptime switch on `build_options.gpu_kind` picks `metal.zig` or `cuda.zig`, and the unselected provider "is parsed but never semantically analyzed, so it costs nothing and needs none of its target's libraries" (docs/REFERENCE.md §9.1) — the CUDA module is fully inert on macOS builds and vice versa. Note also what the exhaustive `switch` buys: adding a new enum member to `backend_kind` makes *every* dispatch site a compile error until it is handled. A plugin registry would have failed at runtime; this fails at build time, everywhere at once.

The build options that shape the backend (the full list is in `docs/REFERENCE.md` §9.1):

| Option | Values | Default | Effect |
|---|---|---|---|
| `-Dbackend` | `native`, `scalar`, `cpu` | `native` | backend implementation; `cpu` = deprecated alias for `scalar` |
| `-Dblas` | `none`, `accelerate`, `openblas`, `mkl`, `blis`, `nvpl`, `blas` | `accelerate` on macOS, else `none` | CBLAS provider for large f32 GEMM; `none` selects the pure-Zig blocked packed GEMM |
| `-Dmax-threads` | 1–64 | 8 | comptime worker-team ceiling *and* runtime default team size |
| `-Dgpu` | `none`, `metal`, `cuda` | `none` | GPU GEMM offload provider |

One consequence of build-time specialization deserves bold text, because it bites people. Without `-Dtarget`, Zig compiles for the host CPU with its **full** feature set — as if `-march=native` were always on — and Fucina's kernels specialize accordingly: NEON/dotprod arms on Apple Silicon, AVX2/AVX-VNNI on modern x86, *unused arms not in the binary*. Two rules follow (README.md, "Builds are tuned to the machine that compiles them"): run the binary on the machine you built it on, and if you must cross-compile, pass `-Dcpu` too (e.g. `-Dtarget=x86_64-linux -Dcpu=x86_64_v3`) — a bare `-Dtarget` gets that architecture's *baseline* features and silently loses the fast kernels. The same rule is a benchmark-validity rule in `docs/BENCHMARK.md`: "A cross-built baseline binary benchmarks the wrong kernels."

The facade (`src/fucina.zig`) re-exports the resulting build facts as comptime constants — `active_backend_kind`, `native_uses_blas`, `native_blas_kind`, `supports_q4_k_mmla`, and friends — so application code can branch on them at zero cost; the branches fold away.

## 6.3 The kernel contract: small, unchecked, allocation-free

Here is the entire state of the dispatch struct (from `src/backend.zig:126–143`):

```zig
pub const Backend = struct {
    pub const kind = active_kind;
    // Atomic: kernels may dispatch on other threads (e.g. dot-backward's
    // OneShotWorker) while a lazy tryWorkPool retry publishes the pool.
    // release/acquire so a racing first observer also sees Pool.init's writes.
    parallel_pool: std.atomic.Value(?*thread.Pool) = .init(null),

    pub fn init() Backend {
        return .{};
    }

    pub fn setWorkPool(self: *Backend, pool: ?*thread.Pool) void {
        self.parallel_pool.store(pool, .release);
    }

    fn parallelConfig(self: *const Backend) ParallelConfig {
        return .{ .pool = self.parallel_pool.load(.acquire) };
    }
```

One field. The backend owns numeric kernels and *nothing else*: no allocator, no configuration blob, no device handles. The single field is an atomic pointer to the worker team, published by the execution context (`Chapter 5`'s `Runtime`) and snapshotted per call. Even the memory ordering carries its reason as a comment — a kernel may be dispatched from another thread while the pool is being lazily created, so the store is `release` and the load `acquire`.

> **Zig note** — `std.atomic.Value(?*thread.Pool)` is an optional pointer wrapped in an atomic cell. Zig's optionals of pointers are guaranteed pointer-sized (null is the zero address), so this is one machine word with compare-and-swap semantics — no lock, no indirection.

Below this struct sit roughly ninety kernel entry points. Their **naming encodes the checking tier**, with one caveat you must internalize:

- `...Into(out, ...) !void` — validates shapes itself and returns `TensorError.ShapeMismatch` on disagreement. This holds for the elementwise, reduction, dot, and matmul families (`addInto`, `sumInto`, `dotInto`, `matmulInto`, …).
- **Caveat:** the conv/pool/norm families (`conv2dInto`, `pool2dInto`, `groupNormInto`, `im2colInto`, `snakeInto`, …) are `...Into`-*named* but plain `void` and **unchecked** — the exec layer validates their geometry before calling them. The naming convention is a strong hint, not a guarantee; `docs/REFERENCE.md` §9.2 lists exactly which families check.
- `...IntoUnchecked` / `...SliceUnchecked` — `void`; the caller has already validated shape and contiguity. Passing wrong geometry is illegal: an out-of-bounds slice panic in safe builds, **undefined behavior in ReleaseFast**.

That last clause is the deal the whole architecture rests on. Kernels get to be small, branch-free, and fast *because* they check nothing — and they get to check nothing because exactly one layer above them (`ExecContext`, Chapter 5) checks everything, once. Validation is not sprinkled defensively through the stack; it has an address.

The allocation contract is equally strict (`docs/REFERENCE.md` §9.2):

- Output buffers are always supplied by the caller. No backend kernel allocates a tensor.
- The vector compute leaves (`src/backend/vector/*`) and the quantized dot kernels are **allocation-free**, full stop.
- One deliberate, documented exception: the quantized-RHS matmul dispatch tier takes an allocator for per-call activation-quantization scratch (Chapter 11's territory), with a stack fast path sized so decode-shaped calls allocate nothing.
- A kernel never creates threads and never assumes a pool exists — `ParallelConfig{ .pool = null }` runs serially. Thread-pool ownership belongs to the execution context.

If you have read [Chapter 10](10-the-guitar-amp.md)'s preview in the course index — a neural amp running inside a real-time audio callback — you can already see why this contract is not stylistic preference. "No allocation, no locks in the leaf kernels, caller-owned buffers" is precisely the set of properties that make a kernel safe to call from a context that must never block.

## 6.4 The scalar backend is the specification

Before admiring any SIMD, meet the code that keeps it honest. This is the native backend's matmul referee — the scalar backend's entire 2-D matmul, from `src/backend/cpu.zig:1295–1318`:

```zig
pub fn matmul2DIntoUncheckedWithConfig(
    out: *Tensor,
    a: *const Tensor,
    b: *const Tensor,
    m: usize,
    n: usize,
    k: usize,
    config: ParallelConfig,
) void {
    _ = config;
    const ad = contiguousDataConst(a, m * k);
    const bd = contiguousDataConst(b, k * n);
    const cd = contiguousData(out, m * n);

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0;
            for (0..k) |p| {
                acc += ad[i * k + p] * bd[p * n + j];
            }
            cd[i * n + j] = acc;
        }
    }
}
```

Three loops, in the order the mathematical definition suggests. Note the signature: it is *identical* to the native backend's — it even accepts the `ParallelConfig`, then ignores it (`_ = config;`), because every scalar kernel is serial. Interchangeable signatures are the point: the two backends are drop-in replacements for each other, differing only in how fast they get the same answer.

The judgment happens in `src/backend/parity_test.zig`, and its design carries three lessons.

**Lesson 1: import both, always.** The parity suite imports `cpu.zig` *and* `native.zig` directly, independent of `-Dbackend` — so `zig build test` always runs the cross-backend comparison, no matter which backend the build selected. The oracle cannot be accidentally compiled out.

**Lesson 2: adversarial sizes.** The elementwise cases run over lengths (`parity_test.zig:16`):

```zig
const elementwise_sizes = [_]usize{ 1, 3, 7, 8, 15, 16, 17, 31, 64, 128, 257, 1024 };
```

Look at those numbers with SIMD eyes: they straddle every plausible vector width (4, 8, 16) and every unroll factor, hitting the "one element", "just under a vector", "exactly a vector", "vector plus tail" and "many vectors plus tail" paths. A 300 000-element case (`parity_test.zig:230`) crosses the parallel-split thresholds so the threaded paths are exercised too. Matmul shapes include deliberately awkward primes — `{7, 11, 13}`, `{33, 17, 23}` — plus a `48×192×128` case that reaches the register-tiled kernels.

**Lesson 3: tolerances are semantics.** Elementwise ops must agree within `1e-6` absolute — one add is one add; SIMD does not change it. But reductions get a *scaled* tolerance (`parity_test.zig`, comment above the check):

```zig
// Scaled tolerance because both backends accumulate n values; the
// SIMD pairwise/parallel summation diverges from a serial loop.
const tol = tolerance * @as(f32, @floatFromInt(n));
```

`sumInto`/`dotInto` agree within `1e-6·n`; the matmul family within `1e-5·k`. Why? Floating-point addition is not associative. A serial sum computes `(((x₀+x₁)+x₂)+x₃)…`; a SIMD sum with four accumulator registers computes four interleaved partial sums and combines them at the end. Both are valid f32 computations; they round differently. The tolerance scaling *is* the numerical model: error grows with the number of accumulated terms.

> **ML note** — This split — *bit-identical* where the operation count and order are preserved, *tolerance-equivalent* where the fast path reassociates — recurs all through numerical computing, and Fucina states it explicitly per kernel family rather than hand-waving "floating point is approximate". Keep the two categories separate in your head; §6.7 returns to them for threading.

Here is the whole philosophy as a build-it-yourself exercise. This is **course code** (not repo code; compile-checked with `zig test` on Zig 0.16.0) — a miniature of `parity_test.zig` that referees a loop-reordered matmul against the naive one:

```zig
const std = @import("std");

/// The referee: the definition of matmul, written to be obviously right.
fn matmulNaive(c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0;
            for (0..k) |p| acc += a[i * k + p] * b[p * n + j];
            c[i * n + j] = acc;
        }
    }
}

/// The candidate: (i, p, j) order — the inner loop walks B's row p and
/// C's row i contiguously, so the auto-vectorizer can do its job.
fn matmulReordered(c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
    @memset(c[0 .. m * n], 0);
    for (0..m) |i| {
        for (0..k) |p| {
            const aip = a[i * k + p];
            const b_row = b[p * n ..][0..n];
            const c_row = c[i * n ..][0..n];
            for (c_row, b_row) |*cj, bj| cj.* += aip * bj;
        }
    }
}

test "reordered matmul agrees with the referee within 1e-5 * k" {
    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();
    const cases = [_][3]usize{ .{ 1, 1, 1 }, .{ 2, 3, 5 }, .{ 7, 11, 13 }, .{ 33, 17, 23 }, .{ 64, 64, 64 } };
    const allocator = std.testing.allocator;
    for (cases) |dims| {
        const m, const n, const k = dims;
        const a = try allocator.alloc(f32, m * k);
        defer allocator.free(a);
        const b = try allocator.alloc(f32, k * n);
        defer allocator.free(b);
        const want = try allocator.alloc(f32, m * n);
        defer allocator.free(want);
        const got = try allocator.alloc(f32, m * n);
        defer allocator.free(got);
        for (a) |*v| v.* = rng.float(f32) * 2 - 1;
        for (b) |*v| v.* = rng.float(f32) * 2 - 1;
        matmulNaive(want, a, b, m, n, k);
        matmulReordered(got, a, b, m, n, k);
        const tol = 1e-5 * @as(f32, @floatFromInt(k));
        for (want, got) |w, g| try std.testing.expect(@abs(w - g) <= tol);
    }
}
```

Write the referee first. Then every optimization you attempt for the rest of your life gets a free correctness proof. This is what makes SIMD *safe to write*: the fast kernel can be as clever as it likes, because a boring judge with adversarial inputs is always in the room. When the blocked packed kernel of §6.6 was added — a new tier coexisting with the row kernels behind a measured dispatch gate — the same philosophy judged it: `src/backend/vector/gemm_blocked_tests.zig` referees it against a naive f64 reference across every tail combination, `kc` boundary, and orientation, saying "yes, still a matmul".

## 6.5 `@Vector`: SIMD as a language feature

Most languages reach SIMD through vendor intrinsics — `_mm256_fmadd_ps` on x86, `vfmaq_f32` on ARM — which means two codebases, or a third-party abstraction layer. Zig builds the abstraction into the language: `@Vector(N, f32)` is a first-class type on which arithmetic operators are SIMD instructions, and the standard library will tell you the right `N` for your target.

Fucina's entire vector-width policy is two lines (`src/backend/vector/common.zig:24–25`):

```zig
pub const vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;
pub const Vf32 = @Vector(vector_len, f32);
```

On NEON that is 4 lanes; on AVX2, 8; on AVX-512, 16; on a target with no SIMD at all, the `orelse 4` fallback still compiles (the operations legalize to scalar code). Because `vector_len` is a `comptime_int`, every loop bound, unroll factor, and tail condition specializes at compile time. One source; per-target machine code; unused ISA arms not in the binary.

Here is the smallest real kernel you can write with it — **course code**, compile-checked:

```zig
const std = @import("std");

const vector_len: comptime_int = std.simd.suggestVectorLength(f32) orelse 4;
const Vf32 = @Vector(vector_len, f32);

fn vecAdd(z: []f32, x: []const f32, y: []const f32) void {
    var i: usize = 0;
    while (i + vector_len <= z.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        const yv: Vf32 = y[i..][0..vector_len].*;
        z[i..][0..vector_len].* = xv + yv;
    }
    while (i < z.len) : (i += 1) z[i] = x[i] + y[i];
}
```

> **Zig note** — Unpack the loading idiom `x[i..][0..vector_len].*`, because it appears hundreds of times in `src/backend/vector/`. `x[i..]` is a slice from position `i` (runtime length). `[0..vector_len]` with a *comptime-known* length re-slices it into a **pointer to an array**, `*const [vector_len]f32` — the length has moved into the type. Dereferencing with `.*` yields the array by value, and a fixed-size array coerces to `@Vector(vector_len, f32)`. The whole chain compiles to a single vector load; the store direction (`z[i..][0..vector_len].* = v`) is a single vector store. And `xv + yv` on vector operands *is* the SIMD add — `+`, `*`, `@min`/`@max`, comparisons, `@mulAdd` all operate lane-wise on `@Vector` types.

The production version adds one more idea. Here is the real `vecAdd`, verbatim (from `src/backend/vector/primitives.zig:52–74`):

```zig
pub inline fn vecAdd(z: []f32, x: []const f32, y: []const f32) void {
    var i: usize = 0;
    while (i + 4 * vector_len <= z.len) : (i += 4 * vector_len) {
        const x0: Vf32 = x[i..][0..vector_len].*;
        const y0: Vf32 = y[i..][0..vector_len].*;
        const x1: Vf32 = x[i + vector_len ..][0..vector_len].*;
        const y1: Vf32 = y[i + vector_len ..][0..vector_len].*;
        const x2: Vf32 = x[i + 2 * vector_len ..][0..vector_len].*;
        const y2: Vf32 = y[i + 2 * vector_len ..][0..vector_len].*;
        const x3: Vf32 = x[i + 3 * vector_len ..][0..vector_len].*;
        const y3: Vf32 = y[i + 3 * vector_len ..][0..vector_len].*;
        z[i..][0..vector_len].* = x0 + y0;
        z[i + vector_len ..][0..vector_len].* = x1 + y1;
        z[i + 2 * vector_len ..][0..vector_len].* = x2 + y2;
        z[i + 3 * vector_len ..][0..vector_len].* = x3 + y3;
    }
    while (i + vector_len <= z.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        const yv: Vf32 = y[i..][0..vector_len].*;
        z[i..][0..vector_len].* = xv + yv;
    }
    while (i < z.len) : (i += 1) z[i] = x[i] + y[i];
}
```

This *three-tier loop* — a 4×-unrolled vector body, a 1× vector body, a scalar tail — is the idiom repeated across every primitive in the file. The unroll gives the CPU four independent load/add/store chains per iteration to overlap; the middle loop mops up remaining full vectors; the scalar tail handles the last `len mod vector_len` elements. On an M1 this compiles to NEON `fadd.4s`; on an AVX2 machine, to `vaddps` on `ymm` registers; on both, from the same eleven-line body.

For an *addition*, unrolling mostly helps the memory pipeline. For a *reduction*, it changes the algorithm. Consider the dot product (`src/backend/vector/primitives.zig:475–496`):

```zig
pub inline fn vecDot(x: []const f32, y: []const f32) f32 {
    if (x.len == 0) return 0;
    var acc0: Vf32 = @splat(0);
    var acc1: Vf32 = @splat(0);
    var acc2: Vf32 = @splat(0);
    var acc3: Vf32 = @splat(0);
    var i: usize = 0;
    while (i + 4 * vector_len <= x.len) : (i += 4 * vector_len) {
        acc0 += @as(Vf32, x[i..][0..vector_len].*) * @as(Vf32, y[i..][0..vector_len].*);
        acc1 += @as(Vf32, x[i + vector_len ..][0..vector_len].*) * @as(Vf32, y[i + vector_len ..][0..vector_len].*);
        acc2 += @as(Vf32, x[i + 2 * vector_len ..][0..vector_len].*) * @as(Vf32, y[i + 2 * vector_len ..][0..vector_len].*);
        acc3 += @as(Vf32, x[i + 3 * vector_len ..][0..vector_len].*) * @as(Vf32, y[i + 3 * vector_len ..][0..vector_len].*);
    }
    while (i + vector_len <= x.len) : (i += vector_len) {
        const xv: Vf32 = x[i..][0..vector_len].*;
        const yv: Vf32 = y[i..][0..vector_len].*;
        acc0 += xv * yv;
    }
    var s = @reduce(.Add, acc0 + acc1 + acc2 + acc3);
    while (i < x.len) : (i += 1) s += x[i] * y[i];
    return s;
}
```

Why *four* accumulators instead of one? A fused multiply-add has a latency of several cycles, and with a single accumulator each FMA must wait for the previous one to finish — the dependency chain serializes the loop at one FMA per `latency` cycles. Four independent accumulators give the out-of-order core four chains to interleave, keeping the FMA units busy every cycle. This is **instruction-level parallelism**, the third axis of CPU performance after SIMD width and thread count.

And now you can see precisely *why* reductions are only tolerance-equivalent to the scalar referee: those four accumulators sum `x` in four interleaved subsequences before a final `@reduce(.Add, …)` combines them. Different association, different rounding, `1e-6·n` tolerance. The design decision and its test consequence are the same fact seen from two sides.

> **Zig note** — Two more builtins from that snippet: `@splat(0)` broadcasts a scalar into every lane (the result type is inferred from context — Zig 0.16 needs no length argument), and `@reduce(.Add, v)` folds a vector to a scalar with the given operator. With `@select` (lane-wise conditional), `@bitCast` (reinterpret f32 lanes as u32 lanes), and `@mulAdd` (fused multiply-add), you have essentially the whole portable-SIMD vocabulary.

How far can portable `@Vector` code go? All the way to transcendentals. Softmax — attention's beating heart — is dominated by `exp`, and calling `libm`'s scalar `expf` per element would strangle it. Fucina's `vexpf` (`src/backend/vector/primitives.zig:364–403`) is a branch-free polynomial exponential in pure `@Vector` code; its doc comment is the algorithm:

```zig
/// SIMD polynomial expf (the ggml_v_expf / ARM optimized-routines scheme):
/// n = round(x*log2(e)) via the 0x1.8p23 shift trick, two-step Cody-Waite
/// reduction r = x - n*ln2, degree-4 polynomial for e^r - 1, then scale by 2^n
/// through the exponent bit field. ...
/// Relative error < 2e-6 over [-87, 88]. No tables, no allocation.
pub inline fn vexpf(comptime W: usize, x: @Vector(W, f32)) @Vector(W, f32) {
    const Vec = @Vector(W, f32);
    const VecU = @Vector(W, u32);
    const xc = @min(@max(x, @as(Vec, @splat(-104.0))), @as(Vec, @splat(89.0)));
    const shift: Vec = @splat(0x1.8p23);
    const z = @mulAdd(Vec, xc, @as(Vec, @splat(0x1.715476p+0)), shift);
    const n = z - shift;
    const r = @mulAdd(Vec, n, @as(Vec, @splat(-0x1.7f7d1cp-20)), @mulAdd(Vec, n, @as(Vec, @splat(-0x1.62e4p-1)), xc));
    const e = @as(VecU, @bitCast(z)) << @splat(23);
```

Hex float literals (`0x1.715476p+0` is log₂e to f32 precision), a `@bitCast` from float lanes to integer lanes so the result's exponent field can be built by a shift, and a final NaN-propagation step via `@select(f32, x != x, x, result)` — NaN is the only value that differs from itself, so `x != x` is a per-lane NaN detector. Every softmax in every transformer this library runs goes through these dozen lines.

The unary-op family generalizes the pattern with comptime dispatch: `vecUnary(comptime op: ops.UnaryOp, z, x)` switches over a comptime enum (`src/backend/ops.zig` — 27 members, `relu` through `reciprocal`, several carrying their provenance as doc comments: `gelu_quant` reproduces ggml's f16-LUT GELU bit-for-bit for llama.cpp parity; `softcap_15` comes from nanochat). One dispatch function, N ops, zero runtime branching — the `switch` on a comptime parameter melts into a direct call per instantiation.

## 6.6 GEMM: the op that pays the bills

Everything in this section refines one function: `C = A·B`. The journey — naive loops → loop reordering → register tiling → cache blocking → provider seams — recapitulates fifty years of dense linear algebra, and Fucina's source documents each step where it happens.

### Loop order: the free 10×

The scalar referee's `(i, j, p)` order is the definition, but look at its memory behavior: the inner loop reads `bd[p * n + j]` with `p` varying — a stride of `n` floats. Every B access is a cache miss waiting to happen, and no SIMD unit can load "every n-th float" efficiently. The fix costs nothing but thought, and the production kernel wears it as a comment (`src/backend/vector/gemm.zig:72–76`):

```zig
// C[i, j] = sum_p A[i, p] * B[p, j]. The natural inner order (i, j, p) reads
// B strided in p, which kills vectorization. Reorder to (i, p, j): broadcast
// A[i, p] as a scalar, multiply by a contiguous slice of B's row p starting at
// j, and accumulate into C's row i starting at j. Now the inner loop is two
// contiguous reads and one contiguous write — vectorizes cleanly.
```

The `(i, p, j)` inner loop is `c_row += a[i,p] * b_row` — a broadcast-FMA over contiguous memory, exactly the `matmulReordered` you compile-checked in §6.4. The transposed-B variant (`matmulTransB`, the shape of every weight matrix applied to activations) needs no reorder at all (`gemm.zig:187–190`): with B stored `[n, k]`, both `A`'s row `i` and `B`'s row `j` are contiguous in `p`, so each output element is a textbook two-stream `vecDot`. And a third orientation story exists for decode: when `m` is tiny (one token), the column-tiled kernel makes the *column* tile the outer loop so B streams through the cache once in total instead of once per row — the comment at `gemm.zig:928–932` records that at `m = 8` the row-outer order cost 8× the memory traffic. Loop order is not a compiler detail; it is the algorithm.

> **ML note** — Those three orientations (NN, TN, NT) are not academic: a linear layer's forward pass is one of them, and the two gradients backprop needs (Chapter 7) are precisely the other two. A GEMM library that only did NN would transpose — i.e. copy — matrices all day.

### Register tiling: reuse before you re-load

The reordered kernel still reads B's row `p` once per output row — `m` times in total. The next insight: load a piece of B into registers *once* and use it for several rows of A. This is `gemmNNRows8` (`src/backend/vector/gemm.zig:1486–1509`), the 8-row register tile:

```zig
inline fn gemmNNRows8(cd: []f32, ad: []const f32, bd: []const f32, row: usize, n: usize, k: usize) void {
    var j: usize = 0;
    while (j + 2 * vector_len <= n) : (j += 2 * vector_len) {
        var acc: [8][2]Vf32 = undefined;
        inline for (0..8) |r| {
            acc[r][0] = @splat(0);
            acc[r][1] = @splat(0);
        }

        for (0..k) |p| {
            const b0: Vf32 = bd[p * n + j ..][0..vector_len].*;
            const b1: Vf32 = bd[p * n + j + vector_len ..][0..vector_len].*;
            inline for (0..8) |r| {
                const a: Vf32 = @splat(ad[(row + r) * k + p]);
                acc[r][0] += a * b0;
                acc[r][1] += a * b1;
            }
        }

        inline for (0..8) |r| {
            cd[(row + r) * n + j ..][0..vector_len].* = acc[r][0];
            cd[(row + r) * n + j + vector_len ..][0..vector_len].* = acc[r][1];
        }
    }
```

Two B vectors are loaded per `p` and reused across **eight** rows of A: sixteen FMAs per two loads. The `acc` array is not really an array — `inline for` unrolls the loop at compile time, so `acc[r][v]` become sixteen named vector *registers* the whole `k`-loop long.

> **Zig note** — This is the second life of `inline for`: register allocation. A runtime loop over `acc` would force the array into memory (you can't index registers). An `inline for` with comptime bounds unrolls into straight-line code where every `acc[r][v]` is a distinct SSA value the compiler assigns to a register. The pattern "small comptime-sized accumulator array + `inline for`" *is* how you write a GEMM microkernel in Zig.

### The memory cliff, and cache blocking

Register tiling holds until the operands stop fitting in cache. The row kernels re-stream all of B for every 8-row block of C; while B lives in L2 that is cheap, and beyond it the kernel becomes a memory benchmark. The numbers are recorded at the dispatch gate (`src/backend/vector/gemm_blocked.zig:104–117`, M1 Max, `-Dblas=none`, ReleaseFast, 8 threads, cool):

> cool-state row kernels peak at ~316 GF/s while operands stay L2-resident and beat the blocked kernel up to 512³ (297 GF/s, 0.94×); by 640³ the blocked kernel wins decisively (374 vs 316, and 2–6× once operands spill: 1024³ ~600 vs ~304, 2048³ ~610 vs ~102).

From 316 down to 102 GF/s purely by growing the matrices — that cliff is arithmetic intensity made visible, and it is the cleanest compute-bound-versus-memory-bound lesson in the repository. The cure is the classic BLIS-style blocked, packed GEMM, and the module doc of `gemm_blocked.zig:1–15` states it exactly:

```zig
//! Cache-blocked, packed (BLIS-style) f32 GEMM for the pure-Zig path.
//!
//! The register-tiled row kernels in gemm.zig re-stream the full B operand
//! for every row block of C, so once the operands exceed L2 the large
//! (training-shaped) GEMMs turn memory-bound. This module implements the
//! classic three-level blocked loop nest with packed panels:
//!
//!   jc (nc) -> pc (kc) -> ic (mc), where
//!     B~(kc x nc) is packed once per (jc, pc) into column panels of width nr,
//!     A~(mc x kc) is packed once per (ic, pc) into row panels of height mr,
//!     and an mr x nr register microkernel streams both panels unit-stride.
//!
//! Orientations: TransA/TransB are absorbed by the packing — the pack loops
//! read the transposed source layout but emit the same panel layout, so the
//! microkernel is orientation-agnostic and NN/TN/NT all share it.
```

*Packing* copies a block of each operand into a scratch buffer whose layout matches exactly the order the microkernel will read — so the innermost loop only ever sees unit-stride streams, regardless of the original layouts or transpositions. The microkernel itself is the whole classical BLAS core in ~35 lines (`gemm_blocked.zig:306–343`):

```zig
fn microKernel(
    comptime accumulate: bool,
    c: []f32,
    ldc: usize,
    a_panel: []const f32,
    b_panel: []const f32,
    kc: usize,
) void {
    var acc: [mr][nr_vecs]Vf32 = undefined;
    inline for (0..mr) |r| {
        inline for (0..nr_vecs) |v| acc[r][v] = @splat(0);
    }

    var p: usize = 0;
    while (p < kc) : (p += 1) {
        var b: [nr_vecs]Vf32 = undefined;
        inline for (0..nr_vecs) |v| {
            b[v] = b_panel[p * nr + v * vector_len ..][0..vector_len].*;
        }
        inline for (0..mr) |r| {
            const a: Vf32 = @splat(a_panel[p * mr + r]);
            inline for (0..nr_vecs) |v| {
                acc[r][v] = @mulAdd(Vf32, a, b[v], acc[r][v]);
            }
        }
    }

    inline for (0..mr) |r| {
        inline for (0..nr_vecs) |v| {
            const dst = c[r * ldc + v * vector_len ..][0..vector_len];
            if (accumulate) {
                dst.* = @as(Vf32, dst.*) + acc[r][v];
            } else {
                dst.* = acc[r][v];
            }
        }
    }
}
```

(`comptime accumulate: bool` compiles two variants — first `pc` block overwrites C, later blocks add — from one body.) The tile shape `mr × nr` is not guessed; it is *budgeted*, and the budget is a comment (`gemm_blocked.zig:52–60`):

```zig
// Register math:
//   aarch64 NEON (32 x 128-bit v-regs, vector_len = 4): mr = 8, nr = 12 ->
//     24 four-wide accumulators + 3 B vectors + 1 A broadcast = 28 live regs.
//   x86-64 AVX2 (16 ymm, vector_len = 8) and other targets: mr = 6,
//     nr = 2 * vector_len -> 12 accumulators + 2 B vectors + 1 broadcast = 15.
pub const mr: usize = if (builtin.cpu.arch.isAARCH64()) 8 else 6;
pub const nr_vecs: usize = if (builtin.cpu.arch.isAARCH64()) 3 else 2;
```

Twenty-eight live registers of NEON's thirty-two; fifteen of AVX2's sixteen `ymm`. Spill one accumulator to the stack and the kernel's throughput collapses — this arithmetic is the difference between a GEMM and a wish. The cache-blocking factors get the same treatment: `kc = 128` on aarch64 keeps the A-panel L1-resident on an M1 P-core; `kc = 512` on Raptor Lake, tuned by an interleaved sweep, "won or tied every shape tested — 253-row NT prefill 850→935 GF/s, 2048³ NN 767→788" (`gemm_blocked.zig:64–77`). Every constant in this file carries its provenance.

One design decision deserves attention because it is a *trade written down* (`gemm_blocked.zig:17–28`): backend kernels must stay allocation-free and the dense matmul entries are infallible (no allocator to thread through), so the pack panels live in a comptime-bounded **static BSS workspace** guarded by a mutex — concurrent blocked GEMMs serialize on it, "a known, accepted trade. If training diamonds ever show this lock hot, bench a per-caller workspace before reaching for anything fancier." Relatedly, `gemmBlockedWithParams` **panics** — deliberately not `std.debug.assert` — on out-of-bounds block params, because the bench sweep feeds runtime params in ReleaseFast, where asserts vanish (`gemm_blocked.zig:154–162`). Safety decisions keyed to build modes, made explicit.

The payoff, as recorded in `docs/BENCHMARK.md` ("Fucina-only kernel context", M1 Max, no llama.cpp pairing): the blocked kernel took 2048³ from **109 to 608 GFLOP/s (5.6×)** on the `-Dblas=none` build, "reaching ~26–35% of Accelerate/AMX on the same machine — what makes the no-BLAS builds credible at training shapes."

### Precision policy is a per-ISA decision

Reduced-precision GEMM makes the ISA differences unhideable, and Fucina's answer is a documented *policy* rather than a shrug. The comment block at `src/backend/vector/gemm.zig:46–56` explains that on aarch64 NEON, f16×f16 `@mulAdd` compiles to native `fmla.8h` — double the f32 lane throughput — so half-precision accumulation is the fast path there and its output is "bit-stable across releases" (`docs/REFERENCE.md` §9.5's summary of the policy). Every other ISA (x86-64 without AVX512-FP16 in particular) legalizes f16 vector arithmetic by promoting through f32 and rounding back *per operation*, which is catastrophic; those targets instead take widened twins that convert each f16 load once and accumulate in f32 — "strictly more accurate, and different from the aarch64 bit pattern" (§9.5 again). Same source tree, two deliberately different numerical behaviors, each the right one for its hardware, both written down. The stakes were measured: on the i9-13950HX, the f32-accumulate f16 GEMM took pp1024 from **17.9 to 354 tok/s** (Fucina-only A/B, `docs/BENCHMARK.md`). bf16 gets the complementary treatment — it is literally the top 16 bits of an f32, so widening is an integer shift (`u16 << 16`, `src/backend/vector/primitives.zig`), and `matmulTransB2DIntoUncheckedBf16Rhs` dots f32 activations against a bf16 weight matrix without ever materializing f32 weights.

### BLAS as a provider, not a religion

That "Accelerate/AMX" aside points at the last tier. BLAS — the Basic Linear Algebra Subprograms — was born in the Fortran era as the standard interface numerical code calls for exactly this operation family, organized in three levels: Level 1 for vector–vector work, Level 2 for matrix–vector (GEMV lives here), Level 3 for matrix–matrix, with GEMM as the Level-3 flagship every optimized implementation is built around. The `S`/`D` type prefixes (SGEMM, DGEMM — single and double precision) are Fortran naming heritage, and the reference implementation is Fortran to this day. Platform BLAS libraries behind that interface (Apple's Accelerate with its AMX units, OpenBLAS, MKL, …) embody decades of tuning; refusing them on principle would be vanity. Fucina treats BLAS as an optional *provider* for large f32 GEMM, selected by `-Dblas` — any CBLAS implementation plugs into the same GEMM seam — and consulted inside the native backend's dispatch. The full precedence, verbatim (`src/backend/native.zig:171–194`):

```zig
    if (comptime build_options.use_gpu) {
        if (gpu.shouldUseGpuForRhs(b, m, n, k)) {
            if (gpu.gemmF32Async(.nn, a, b, out, m, n, k)) return;
        }
    }
    if (comptime build_options.use_blas) {
        if (shouldUseBlas(m, n, k)) {
            blasGemm(
                cblas_no_trans,
                cblas_no_trans,
                m,
                n,
                k,
                contiguousDataConst(a, m * k),
                k,
                contiguousDataConst(b, k * n),
                n,
                contiguousData(out, m * n),
            );
            return;
        }
    }
    vector.matmul2DIntoUncheckedWithConfig(out, a, b, m, n, k, config);
```

Read the structure: GPU (if built in) may decline; BLAS (if built in) may decline; pure Zig always answers. The `comptime` guards mean a plain build contains *neither* upper tier — not disabled, absent. Each gate is measured, and the gates cut both ways. `shouldUseBlas` requires all of `m, n, k ≥ 16` (`native.zig:1330–1332`), and a **recorded negative** in `docs/BENCHMARK.md` keeps it honest: lowering the gate to `m ≥ 2` routed small-m tall-skinny NT GEMMs (m ≈ 8, n up to 152681, k = 2048) to Accelerate and *lost* — 37.8 s wall versus 32.5 s for the fixed vector column kernel (M1 Max, 2026-07-07). "The m >= 16 BLAS gate stays; do not re-lower it without new evidence on these shapes." A big-name library is a data point, not an authority.

Within pure Zig, `shouldUseBlocked(m, n, k)` (`gemm_blocked.zig:123–126`) routes work of at least 192 Mi multiply-adds with `m, n ≥ 32, k ≥ 16` to the blocked kernel, chosen from the measured 512³–640³ crossover; everything below stays on the register-tiled row kernels, which are faster there *and* avoid the packing traffic.

Provider choice ripples upward, too. The Winograd convolution route (a transform that computes a 3×3 convolution with ~2.25–4× fewer multiplications, at ~1e-6/1e-5 relative drift — docs/REFERENCE.md §9.6) defaults **on** for `-Dblas=none` builds and **off** when a platform BLAS backs the matmul — Accelerate's AMX prefers one big im2col GEMM over many small tile GEMMs (`winograd_default_on = !native_uses_blas`, docs/REFERENCE.md §9.6). An optimal routing decision three layers up depends on which GEMM engine sits at the bottom; pretending the layers are independent would leave measured performance on the table.

## 6.7 The worker team

One core is not enough for prefill. The naive approach — spawn threads inside each op, join before returning — dies by a thousand syscalls: an LLM decode step is a dense stream of small-to-medium kernels, and thread creation (or even a futex park/wake round trip per op) costs more than many of the kernels themselves. Fucina's answer is a **persistent, bounded worker team** created lazily by the execution context, published to the backend through that one atomic pointer from §6.3, and dispatched via fork-join.

The core API takes a slice of plain task structs and a comptime function (`src/thread.zig:232–270`):

```zig
    // Fork-join parallel-for over a persistent hot team: runs `run(&tasks[i])`
    // for every i in [0, tasks.len), the caller executing chunk 0 and the team
    // the rest, rendezvousing before return. This is the default substrate for
    // splitting a numeric kernel across cores. Unlike `spawnWg`/`waitAndWork`
    // (general async tasks routed through std.Io's executor, which heap-allocs a
    // task node per spawn and parks/wakes each worker via a futex syscall), the
    // team here stays hot between dispatches (spin-then-park), so a dense stream
    // of small ops costs atomics instead of kernel round-trips.
    pub fn parallelChunks(
        self: *Pool,
        comptime Task: type,
        tasks: []const Task,
        comptime run: fn (*const Task) void,
    ) void {
        const n = tasks.len;
        if (n == 0) return;
        if (n == 1) {
            run(&tasks[0]);
            return;
        }
        const barrier = self.ensureBarrier() catch null;
        if (barrier == null or barrier.?.worker_count == 0) {
            for (tasks) |*t| run(t);
            return;
        }
        if (self.parallel_chunks_active.cmpxchgStrong(false, true, .acquire, .monotonic) != null) {
            for (tasks) |*t| run(t);
            return;
        }
        defer self.parallel_chunks_active.store(false, .release);

        const Thunk = struct {
            fn call(ctx: *anyopaque, index: usize) void {
                const base: [*]const Task = @ptrCast(@alignCast(ctx));
                run(&base[index]);
            }
        };
        barrier.?.dispatch(n, @ptrCast(@constCast(tasks.ptr)), Thunk.call);
    }
```

Three things to notice. First, **graceful degradation is the control flow**: no team, team busy, single task, re-entrant call — every failure mode falls back to running the tasks serially on the caller. A kernel that uses `parallelChunks` cannot deadlock and cannot fail; parallelism is strictly an accelerant. Second, the caller **works too** — it executes chunk 0 while the team takes the rest, so an 8-way split uses 7 workers plus the dispatching thread. Third, the comptime `Task`/`run` pair erases to an `*anyopaque` thunk at the barrier layer: generic at the call site, monomorphic and allocation-free underneath.

The "hot" in *hot team* is the spin-then-park worker loop (`src/thread.zig:677–695`):

```zig
        while (true) {
            var spins: u32 = 0;
            var gen = self.generation.load(.acquire);
            while (gen == seen) {
                spins +%= 1;
                if (spins < spin_budget) {
                    std.atomic.spinLoopHint();
                } else {
                    // Park. Announce first (seq_cst, paired with dispatch's
                    // parked check); the futex re-checks the value on entry, so
                    // a wake or wake-skip that raced ahead of the park is not
                    // lost — a stale `gen` makes the kernel refuse to sleep.
                    _ = self.parked.fetchAdd(1, .seq_cst);
                    self.io.futexWaitUncancelable(u32, &self.generation.raw, gen);
                    _ = self.parked.fetchSub(1, .seq_cst);
                    spins = 0;
                }
                gen = self.generation.load(.acquire);
            }
            seen = gen;
```

Workers watch a *generation counter*. A dispatch bumps it; any worker still spinning notices within nanoseconds — no syscall on either side. Only after `spin_budget` empty iterations does a worker park on a futex (one syscall), announcing itself in a `parked` counter first so the dispatcher knows whether a wake syscall is even needed. In a dense op stream, dispatch costs atomics instead of kernel round trips.

`parallelChunks` has a sibling for irregular work: `parallelChained` (`src/thread.zig:286–309`) runs a *dependency graph* over the same hot team — a task makes its successors runnable via `chain.enqueue(i)` — and it comes with a contract sharp enough to draw blood: every index must become runnable **exactly once**. Under-enqueueing never terminates; enqueueing twice corrupts the intrusive Treiber stack that carries runnable indices. The interesting engineering choice is *where* that contract is checked: `const chain_checks = std.debug.runtime_safety;` (`thread.zig:417–424`) compiles the verification machinery into Debug/ReleaseSafe builds — which panic with a precise message on violation — and out of ReleaseFast entirely, zero cost. Between "always check" (too slow for a hot dispatch path) and "trust the caller" (undebuggable), Zig's build modes offer a third door: check in the builds whose job is checking.

How long to spin is a genuinely hard tuning problem, and the source is candid about it (`src/thread.zig:426–444`): the 32768 default "is the measured M1 tuning (long spin throttles the M1 clock)", and a full cool-machine sweep on the x86 box found the response "workload-coupled and U-shaped, so NO static value dominates there either" — 512 iterations makes a speech-encoder workload ~6% faster but costs qwen3 prefill ~6% and decode ~2%; 2048–4096 is "the worst of both"; 262144 regresses the encoder ~5% by starving compute cores with spin power. The resolution is an env knob, `FUCINA_SPIN_BUDGET`, rather than a pretend-universal constant. When measurement says "it depends", the honest engineering answer is a documented knob.

### How many threads?

Bounded, and *physically* bounded. The team ceiling is `-Dmax-threads` (default 8 — the M1 Max P-core count, with the full thermal reasoning in a comment at `src/parallel.zig:5–18`: prefill is fastest at 8 cores cool but ~6 heat-soaked, decode is ~8–14% faster at 6, and no single value wins everywhere, so the default chases best cold prefill and `FUCINA_MAX_THREADS` — mirroring llama.cpp's `-t` — drops it for sustained workloads).

And the sizing logic refuses to count hyperthreads, for a measured reason (`src/parallel.zig:50–54`):

```zig
        // SMT machines double-book cores in the logical count, and an
        // HT-oversubscribed team collapses throughput (i9-13950HX: a
        // 16-worker team pinned to 8 P-cores' hyperthreads ran 19s of
        // prefill in 43s — the x86 threading finding in docs/BENCHMARK.md).
```

Sit with that number: *more* threads made the same work take **2.3× longer** — 19 seconds of prefill became 43. Two hyperthreads share one core's FMA units, so pinning two spinning workers per P-core buys zero extra arithmetic while doubling contention on the barrier. `cpuThreadCount` therefore clamps to the physical-core count (macOS `sysctl hw.physicalcpu`; on Linux a libc-free dedup of sysfs `thread_siblings_list` intersected with the affinity mask), a structural no-op on Apple Silicon, which has no SMT. If you ever needed one exhibit for "thread count is a tuning parameter, not a virtue", this is it.

Whether a given kernel splits at all is *threshold-gated* — parallelism has fixed costs, and small ops don't earn them back. The gates live in `src/backend/vector/common.zig` against tuned constants in `src/parallel.zig`: elementwise kernels stay serial below 256 Ki elements; GEMMs split by rows above 1 Mi multiply-adds; and decode-shaped GEMMs (`m < 32` with `n ≥ 128`) split by *columns* in 64-column chunks instead, because there aren't enough rows to share. That `m < 32` boundary is the prefill/decode divide from §6.1, cast in code.

Finally, determinism — the two categories from §6.4 again, now for threads. Parallel elementwise, conv, pool, and Winograd splits are **bit-identical** to the serial path, because tasks own disjoint output ranges: no value is computed differently, only elsewhere (`docs/REFERENCE.md` §9.4; the blocked GEMM's cell grid gives each C tile exactly one writer, so it is deterministic and thread-count-independent too). Threaded *reductions* and GEMM, by contrast, state a reassociation tolerance, exactly as their SIMD versions do. Fucina never blurs the two — and now you know enough to see that the distinction isn't pedantry: one category can be regression-tested with `==`, the other needs an error model.

> **Zig note** — One platform concession hides in the worker: on macOS, workers *and the dispatcher* pin themselves to `QOS_CLASS_USER_INTERACTIVE` via `extern "c" fn pthread_set_qos_class_self_np` — otherwise the scheduler may demote a spinning worker (or the dispatcher, which computes chunk 0 of every op) to an efficiency core, and a fork-join barrier always runs at its straggler's speed (`src/thread.zig:451–456`). Calling a C API takes one `extern` declaration; no binding generator, no FFI layer.

## 6.8 Down to single instructions

Portable `@Vector` covers almost everything — but "almost" matters when a specific instruction is the whole game. Quantized inference ([Chapter 11](11-model-files-and-quantization.md)) runs on int8 dot products, and modern ISAs have dedicated instructions for them: NEON's `sdot` computes four 4-way i8×i8→i32 dot products in one instruction; x86's `vpdpbusd` (AVX-VNNI / AVX512-VNNI) is its 8-lane cousin. LLVM cannot always be coaxed into emitting these from portable IR — the repo records the verification that a clamp-pattern `@Vector` formulation compiles to the wrong instruction (`src/backend/quant/common.zig:177–180`) — so Fucina hand-places them with inline assembly, each behind a comptime gate, each with a portable twin (`quant/common.zig:483–494`):

```zig
pub fn sdotI8x16(acc: QKV4i32, a: QKV16i8, b: QKV16i8) QKV4i32 {
    if (comptime builtin.cpu.arch == .aarch64) {
        var out = acc;
        asm ("sdot %[out].4s, %[a].16b, %[b].16b"
            : [out] "+w" (out),
            : [a] "w" (a),
              [b] "w" (b),
        );
        return out;
    }
    return sdotI8x16Portable(acc, a, b);
}
```

The pattern is the point: the asm arm and the portable arm are **proven bit-equal by an in-tree test** (the ints make exact equality checkable — no tolerance needed), so every target compiles, every target is testable, and the fast arm is a verified drop-in. The feature gates are comptime target queries — `std.Target.aarch64.featureSetHas(builtin.cpu.features, .i8mm)` guards `smmla`, with a comment noting that M1-class cores are aarch64 but *lack* FEAT_I8MM and take the portable path.

This layer is also where real-world ISA archaeology lives. The x86 `vpdpbusd` arm builds its own mnemonic at comptime (`quant/common.zig:342–354`):

```zig
        // ENCODING IS LOAD-BEARING: LLVM's asm parser does not feature-check
        // inline asm and resolves the bare mnemonic to the EVEX (AVX512-VNNI)
        // form, which SIGILLs on AVX-VNNI-only cores (Alder/Raptor Lake) —
        // their VEX form must be selected with the explicit {vex} prefix
        // (LLVM defines the AVX-VNNI aliases as ExplicitVEXPrefix). Cores
        // gated in via AVX512-VNNI+VL (Ice Lake, no AVX-VNNI) take EVEX.
        const mnemonic = comptime if (has_x86_avxvnni) "{vex} vpdpbusd" else "vpdpbusd";
```

The same instruction name has two encodings; the assembler picks the wrong one unless told; the wrong one crashes on exactly the CPUs the arm is *for*. Comptime string concatenation building the assembly text itself is the fix.

Two testing caveats are documented and worth repeating precisely. First: **Debug builds do not execute these asm arms.** Zig's self-hosted x86-64 backend (the Debug default on x86_64-Linux) lacks the newer VEX mnemonics, so the arms are additionally gated on the LLVM backend (`has_llvm_asm`, `quant/common.zig:279–284`) — Debug builds run the exact portable twins; exercising `sdot`/`vpdpbusd` for real requires ReleaseSafe/ReleaseFast. Second: emulators lie — qemu 7.0 "executes AVX2 silently wrong — no SIGILL, corrupt lanes — never validate with it"; qemu ≥ 9.2 is required. The standalone checker `src/x86dot_check.zig` runs kernel-vs-scalar asserts with printable FNV-1a bit checksums so runs from different machines and emulators can be diffed, and its header keeps a dated *execution attestation table* of which arms have actually run on which hardware. Claiming coverage you haven't executed is the same sin as inventing a benchmark number.

The kernels that consume these primitives — block dequantization, K-quant dots, packed weight layouts — belong to [Chapter 11](11-model-files-and-quantization.md) and [Chapter 14](14-the-low-bit-frontier.md); what this chapter owns is the dispatch-and-ISA seam they plug into.

## 6.9 Honest numbers

Everything above ends in a claim of the form "X is faster". This section is about what such a claim is worth, because most performance claims in the wild are worth little. Fucina's benchmark record, `docs/BENCHMARK.md`, opens with its own epistemology:

> **The record is one snapshot, taken as of 2026-07-04** […]
> - **Every number carries its hardware and measurement conditions.** CPU benchmarks on laptops are thermally and page-cache sensitive and shape-specific; a number without its conditions is not a result.
> - **Losses are recorded as plainly as wins.**

The protocol behind any "parity-or-faster" claim (`tools/bench_gate.py`) is deliberately conservative: every row runs in **both process orders** (Fucina→llama.cpp, then llama.cpp→Fucina) so order effects and thermal drift contaminate both sides equally; rows whose coefficient of variation exceeds 8% are reported **NOISY** and *not counted as results*; raw stdout and exact command lines are archived per subprocess; medians are compared, not best cases. Prompt length is treated as a benchmark parameter in its own right — the routine matrix runs twenty lengths (`1,2,3,…,129,256`) chosen to straddle tile and threshold boundaries, because kernels have tile sizes and models don't; quoting `pp4` alone is quoting the tail path. And an entire subsection is devoted to thermal discipline on Apple Silicon: "The single largest source of wrong conclusions in this file's history was chip temperature" — heat soak *inverts thread scaling*, long sweeps depress the rows measured late, and "several apparent 'llama.cpp wins decode' readings evaporated when re-measured cool and prewarmed". Authoritative comparisons are cool, isolated, interleaved A/B pairs with pre-cooldowns.

With the protocol stated, the headline (README.md, snapshot 2026-07-04, Apple M1 Max, CPU-only both sides; `docs/BENCHMARK.md` records the run conditions — 8 threads, llama.cpp with its Accelerate BLAS backend, its default on this platform): of 236 paired sweep cells across Qwen3 dense, Qwen3.5, the 30B MoE, and Gemma-26B, **Fucina is faster in 221 and at parity in 13** — dense prefill geomeans 1.18–1.81× per format, large MoE prefill up to ~2×. On the x86 Raptor Lake box (AVX2+VNNI, no BLAS either side), dense quantized formats show paired-gate medians of 1.32–1.95×.

And the losses, because they are the price of believing the wins:

- **Qwen3.5-0.8B Q8_0 pp32: 0.86×** for Fucina (M1, 3 interleaved rounds, both orders). Confined to that shape — pp128 (1.09×), pp512 (1.17×) and decode (1.37×) all win — with a diagnosed-but-open process-to-process bimodality on the Fucina side.
- **30B Q6_K decode: 0.88×** — recorded *without* page-cache prewarming; its Q5_K_M sibling measures 1.36× under the stricter prewarmed protocol, so the cell is likely conditions-bound, "but it stands until re-measured." That sentence is the whole ethic: an unflattering number is not deleted on suspicion; it is kept with its caveat until a better measurement replaces it.
- **x86 MoE decode: 0.90–0.95×** (GEMV-shaped: weight-bandwidth-bound at m = 1) and **Gemma-26B small-batch prefill pp1–9: 0.85–0.99×**, both llama.cpp wins on the record.

The record also keeps a **"Recorded negatives"** section — optimizations tried, measured, and declined, written down "so they are not re-tried or over-claimed": the small-m BLAS routing from §6.6; a Q5_K decode repack that helped decode but regressed prefill (reverted); a q8_0-quantized KV cache that buys 1.88× context capacity but *slows* M1 decode (attention there is compute-bound, and the dequant adds ~2.3× to the attention phase); residual-add GEMM epilogues whose entire opportunity measured at ≤1.2% of forward time — and which "cannot be proven bit-exact for BLAS", a correctness argument doing performance triage. A graveyard of measured dead ends is as much a performance artifact as the kernels: it stops future contributors from rediscovering losses, and it teaches the reader which intuitions failed.

Treat every figure in this section as what the record says it is: shape- and machine-specific, "measured as of the snapshot date, on that machine". Two CPUs, one thread count each, specific GGUFs, a continuously advancing reference. A ratio quoted without its row is folklore.

## 6.10 GPU offload is a seam, not a backend

One paragraph of GPU, as promised — because architecturally that is what it is. `-Dgpu=metal` (macOS) or `-Dgpu=cuda` (Linux) compiles in an eager GEMM *provider*, selected by the same comptime-switch pattern as everything else in this chapter, and slotted in as the first tier of the dispatch precedence you saw in §6.6. The contract is "gates decide, dispatchers run": cheap shape gates (general GEMM requires `m ≥ 32, n ≥ 32, k ≥ 16`) sit at the dispatch sites, and every dispatch entry returns `false`/`null` when the GPU did not run, with the caller falling through to BLAS or the vector kernels — correctness never depends on the GPU (docs/REFERENCE.md §9.9). Tensors carry no device type and no location state; there is no graph, no placement planner, no general GPU runtime. Metal offloads specific dense and quantized GEMM shapes through vendored MSL kernels (the MLX "steel" GEMM, llama.cpp's `mul_mm`); the CUDA sibling binds `libcuda`/`libcublas` via `dlopen` at runtime, so no CUDA SDK is needed at build time. The ordering and teardown contract lives in `docs/GPU-OFFLOAD.md`, and the inference tricks that lean on offload are [Chapter 13](13-inference-tricks.md)'s material.

It is worth pausing on how un-dramatic this is. Adding GPU support to a framework usually means a device abstraction, a memory manager, a stream scheduler. Here it is a comptime-selected module whose every entry point is allowed to say "no" — thirty lines of seam (`src/backend/gpu.zig`) in front of two provider files, made possible because §6.6's dispatch was *already* a fall-through chain of refusable tiers.

## What you now know

- All of a network's time funnels through backend kernels, and matmul dominates; `m` (tokens in flight) splits the world into prefill-shaped and decode-shaped work.
- Fucina has exactly two backends — `scalar` (the reference) and `native` (the fast one) — selected by a comptime switch; the loser is dead code, and `cpu` is only a deprecated alias for `scalar`.
- The kernel contract: small, unchecked, allocation-free; validation lives once, in the exec layer. `...Into` names *usually* mean self-checking — but the conv/pool/norm families are `...Into`-named and unchecked, and wrong geometry on unchecked entries is UB in ReleaseFast.
- The scalar backend is an executable specification: the parity suite imports both backends regardless of `-Dbackend`, runs adversarial sizes straddling every vector width, and encodes numerics in its tolerances — exact for elementwise, `1e-6·n` / `1e-5·k` where SIMD or threads reassociate.
- `@Vector` + `std.simd.suggestVectorLength` gives one portable SIMD source that compiles to NEON on M1 and AVX2 on x86; the three-tier loop, the `x[i..][0..vector_len].*` idiom, multi-accumulator ILP, and `inline for` as register allocation are the working vocabulary.
- GEMM is a staircase: loop reordering → register tiling → cache-blocked packing (the ~316→~102 GF/s memory cliff, cured to ~610 at 2048³ on the M1 record) → optional BLAS and GPU tiers that may always decline. Every tile size and gate constant in the tree carries its measured provenance.
- Parallelism is a bounded, persistent, spin-then-park team dispatched by `parallelChunks` — atomics instead of syscalls, caller as chunk 0, serial fallback everywhere. Threads are counted physically (the 16-workers-on-8-cores 19 s→43 s collapse), gated by work thresholds, and split so that elementwise/conv/pool/Winograd stay bit-identical while reductions/GEMM state a tolerance.
- Hand-placed instructions (`sdot`, `{vex} vpdpbusd`) live behind comptime feature gates with bit-equal portable twins — and Debug builds run the twins, not the asm.
- A benchmark number without its machine, date, protocol, and caveats is not a result; losses and declined optimizations belong in the record next to the wins.

## Explore the source

- `src/backend.zig` — the two-backend comptime switch, the facade constants, and the ~90-method dispatch struct whose only state is an atomic pool pointer.
- `src/backend/cpu.zig` — the scalar reference; read its matmul first and let everything else in this chapter be judged against it.
- `src/backend/parity_test.zig` — the referee: adversarial sizes, scaled tolerances, both backends imported unconditionally.
- `src/backend/vector/primitives.zig` — `vecAdd`'s three-tier loop, `vecDot`'s four accumulators, `vexpf`'s branch-free exponential.
- `src/backend/vector/gemm.zig` and `src/backend/vector/gemm_blocked.zig` — loop-order comments, the 8-row register tile, and the BLIS-style nest with its register-budget and cache-tuning comments.
- `src/thread.zig` and `src/parallel.zig` — `parallelChunks`, the spin-then-park worker loop, the spin-budget sweep notes, and the physical-core sizing with the SMT collapse number.
- `src/backend/quant/common.zig` — the ISA-gated int8 arms, the `{vex}` encoding trap, and the portable twins.
- `docs/BENCHMARK.md` — the protocol, the scoreboard with its losses, and the recorded negatives.
- `docs/REFERENCE.md` §9 — the full backend reference this chapter is a guided tour of.

## Exercises

1. **(Easy)** Extend the course `vecAdd` snippet with `vecMul` and `vecScale` (multiply by a scalar — `@splat` the scalar once, outside the loop), and add them to the parity test over the same awkward lengths. Exact equality should hold for all three: why does no tolerance appear anywhere in this exercise?
2. **(Medium)** Add the 4×-unrolled tier to the course `vecAdd`, then time both versions over a 16 Mi-element buffer with `std.time.Timer` in a ReleaseFast build (`zig test -OReleaseFast`). Then shrink the buffer to 64 Ki and measure again. Explain the difference using §6.6's compute-bound/memory-bound vocabulary — which resource does `vecAdd` saturate at each size?
3. **(Medium)** Write `dotSimd`, a standalone copy of §6.5's `vecDot`, and `dotOneAcc`, the same kernel with a single accumulator; benchmark both against a plain-loop scalar referee `dotScalar` on a large buffer in ReleaseFast. Estimate your CPU's FMA latency from the 1-accumulator/4-accumulator ratio. Verify that `dotOneAcc` and `dotSimd` do *not* produce bit-identical results on random data, and explain why both still pass a `1e-6·n` parity test.
4. **(Hard)** Add a 4-row register tile to the course `matmulReordered` (load B's row slice once per `p`, reuse it across 4 rows of A, accumulators in an `inline for`-unrolled array), guard the leftover rows with the plain kernel, and prove it against `matmulNaive` with the `1e-5·k` tolerance over the prime-sized shapes. Measure at 512×512×512 in ReleaseFast.
5. **(Hard)** On your own machine, run Fucina's GEMM sweep: `zig build bench-gemm -Dblas=none -- --sweep` (ReleaseFast; read the tuning comments at the top of `src/backend/vector/gemm_blocked.zig` first). Find where the row-kernel/blocked-kernel crossover sits for your cache hierarchy, and compare with the M1 Max and Raptor Lake numbers recorded in the source. If your best `BlockParams` differ from the defaults, work out which cache level each panel is being sized for — then check your reasoning against the `kc`/`mc`/`nc` comment.

---

[Previous: The operation library](05-the-operation-library.md) · [Next: Autograd — the graph hidden in the values](07-autograd.md)
