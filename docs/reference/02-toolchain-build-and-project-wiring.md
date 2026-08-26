# 2. Toolchain, build, and project wiring

## 2.1 Toolchain (`AGENTS.md`, `README.md`)

Fucina is pinned to **Zig 0.16.0** — `zig version` must print `0.16.0`; other
versions do not build. `build.zig.zon` names the package `.fucina` and its
`minimum_zig_version = "0.16.0"` turns an older toolchain into a proper
error (a newer toolchain passes that check but is equally unsupported — the
pin is exact). Every module, executable, and option is wired in
`build.zig`; the manifest has no dependencies of its own. There
is also no C/C++ build system — the only non-Zig translation units are a few
vendored shims (`src/backend/metal/shim.m`, the miniaudio/MIDI shims under
`examples/`) compiled by `build.zig` itself when the relevant option or
example requires them. System dependencies appear only when options select
them: a CBLAS provider for `-Dblas=...`, Apple frameworks for
`-Dgpu=metal`/`-Dblas=accelerate` and the audio examples, libc for
`-Dgpu=cuda` (the CUDA driver and cuBLAS are `dlopen`ed at runtime — no CUDA
SDK at build time), and a Rust toolchain for `-Dllguidance=true` (the
vendored llguidance staticlib is Rust, built via cargo from `build.zig`;
[§2.2](02-toolchain-build-and-project-wiring.md#22-build-options-buildzig)).

```sh
zig version        # 0.16.0
zig build test     # all test roots; no model assets needed
zig build --help   # lists every step and project option below
```

## 2.2 Build options (`build.zig`)

All project options are consumed at **comptime** through the generated
`build_options` module ([§2.4](02-toolchain-build-and-project-wiring.md#24-module-graph-and-options-wiring-buildzig)) — backend dispatch is compiled away, and unused
kernel arms are not in the binary.

| Option | Values | Default | Effect | Constraints |
| --- | --- | --- | --- | --- |
| `-Dbackend` | `native` \| `scalar` | `native` | Kernel implementation set. `native` = Zig SIMD vector kernels + optional BLAS; `scalar` = the reference backend (correctness oracle — native and scalar must agree). | |
| `-Dblas` | `none` \| `accelerate` \| `openblas` \| `mkl` \| `blis` \| `nvpl` \| `blas` | `accelerate` on macOS *targets*; on a **native Linux** build, auto-detected from the linker cache (NVPL on aarch64 / MKL on x86-64, then OpenBLAS, then BLIS; the generic `blas` is never auto-selected) with one stderr line reporting the pick; `none` when cross-compiling or nothing is found | CBLAS provider backing the native backend's large-GEMM arms; `none` keeps the pure Zig vector kernels (including the blocked packed f32 GEMM). | `accelerate` on a non-macOS target **panics the build**. |
| `-Dblas-threads` | `u32` | `0` | Pins the vendor BLAS thread count for explicit providers (OpenBLAS/MKL/BLIS/NVPL); `0` keeps the provider default. | No effect with `-Dblas=none`. |
| `-Dmax-threads` | `usize` | `8` | Comptime worker-team ceiling **and** runtime default thread count (`src/parallel.zig`). Sized for M1 Max P-cores; many-core servers must raise it at build time (`FUCINA_MAX_THREADS` only lowers it at runtime). | Outside 1–64 **panics the build**. |
| `-Dgpu` | `none` \| `metal` \| `cuda` | `none` | GPU GEMM offload provider ([§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)). `metal`: big f32/f16/bf16 GEMMs, dense quantized prefill linears, and the MoE expert FFN on macOS. `cuda`: the same surface plus streaming attention forward and opt-in decode GEMV on Linux/NVIDIA, no SDK at build time. Decode below the work gates and training stay on CPU. | `metal` on a non-macOS target **panics**; `cuda` on a non-Linux target **panics** (cross-compiling from macOS with `-Dtarget=x86_64-linux-gnu` is the supported path). |
| `-Dparakeet-mic` | `bool` | `false` | Links the vendored miniaudio capture stack into the `parakeet` example so `--mic` (live microphone) works; default off keeps the parakeet build fast. | Only affects the parakeet executable/tests. |
| `-Dllguidance` | `bool` | `false` | Builds the vendored [llguidance](../../vendor/llguidance/README.md) constrained-decoding engine (`cargo build` in `vendor/llguidance`) and links its staticlib into the qwen3/gemma4/lmserve examples and the models, serving, lmserve, and snippet-check test roots, enabling `models.text.llguidance` grammar/JSON-schema token masking ([§13.6](13-the-model-stack-fucina_models.md#136-sampling-srcmodelstextsamplerzig)). Off (the default) the build stays pure Zig and `models.text.llguidance.Constraint.init` returns `error.LlguidanceNotEnabled`; the `LogitProcessor` seam itself is always available. | Requires a Rust toolchain >= 1.87 on PATH when enabled. |
| `-Dvector-scan` | `bool` | `false` | Vectorizes the scan kernels (`cumsum`/`cumprod` and cumsum's reverse VJP pass). Off = the documented serial-per-row scans. On: non-last-axis scans vectorize across independent columns (bitwise identical to serial); last-axis scans use an in-register prefix scan — still bitwise deterministic for any thread count, but the accumulation order differs from the serial default (the sum-SIMD-lanes rounding class; exact for integer-valued data). Measured M1 ReleaseFast 256×8192: cumsum 3.3×, cumprod 5.2× (last axis), 4.3× (non-last, bit-identical). |
| `-Doptimize` | `Debug` \| `ReleaseSafe` \| `ReleaseFast` \| `ReleaseSmall` | `Debug` | Standard Zig optimize mode. Build with `ReleaseFast` whenever speed matters (Debug is 10–50× slower); validate in Debug/ReleaseSafe, bench in ReleaseFast. | `x86dot-check` is always built ReleaseSafe regardless. |
| `-Dtarget`, `-Dcpu` | standard queries | host, native CPU | Cross-compilation target and CPU model. | See below — a bare `-Dtarget` silently loses the fast kernels. |

Constraint violations are **build-time panics** (`@panic`/`std.debug.panic`
inside `build()`), not recoverable configuration errors: the panic checks run
against the *target* OS (`target.result.os.tag`), not the host, so
`-Dgpu=cuda -Dtarget=x86_64-linux-gnu` from macOS builds fine while
`-Dgpu=cuda` alone on macOS panics.

**CPU targeting is native by default.** With no `-Dtarget`, Zig targets the
compiling machine's exact CPU (full detected feature set, like
`-march=native`), and the kernels' comptime feature gates
(`src/backend/quant/common.zig`) compile in the matching arms — NEON/sdot on
Apple Silicon, AVX2/AVX-VNNI on modern x86, smmla on I8MM-class ARM servers,
portable vectors elsewhere. Unused arms are compiled out entirely; there is
no runtime dispatch. Cross-compiling with `-Dtarget=...` drops to that
architecture's *baseline* unless `-Dcpu=...` names a model (`x86_64_v3`,
`alderlake`, `znver4`, `neoverse_v1`, …). Two rules follow: build on the
machine that will run the binary, or pin `-Dcpu` to match it.

The resolved configuration is visible on the `fucina` module root as
comptime constants (`active_backend_kind`, `native_blas_kind`,
`native_uses_blas`, `native_uses_accelerate`, `native_blas_threads`,
`parallel.vector_max_threads`):

```zig
const std = @import("std");
const fucina = @import("fucina");

test "build options are comptime facts on the module root" {
    // Baked in by build.zig's `build_options`; all comptime-known.
    const kind: fucina.BackendKind = fucina.active_backend_kind; // -Dbackend
    try std.testing.expect(kind == .native or kind == .scalar);
    if (fucina.native_uses_blas) // -Dblas != none
        try std.testing.expect(fucina.native_blas_kind != .none);
    try std.testing.expect(fucina.parallel.vector_max_threads >= 1); // -Dmax-threads
}
```

Note `fucina.BackendKind` has two members (`scalar`, `native`): build.zig
bakes the raw three-member `-Dbackend` value (including the deprecated
`cpu`) into `build_options.backend_kind`, and the `cpu → .scalar` mapping
happens at file scope of `src/backend.zig`. At runtime the effective worker
count never exceeds the comptime ceiling — `fucina.parallel.setMaxThreads(n)`
is the programmatic counterpart of `FUCINA_MAX_THREADS` (mirrors llama.cpp's
`-t`; call once at startup, before the first parallel op — the first
`cpuThreadCount` call latches the value). The two are not identical: the env
var only *lowers* the detected CPU count, while `setMaxThreads` *replaces*
it and can raise the team size above the detected count, up to the ceiling
([§6.6](06-the-execution-runtime-execcontext-and-the-memory-model.md#66-the-worker-team-srcthreadzig-srcparallelzig)):

```zig
test "runtime worker count never exceeds the comptime ceiling" {
    fucina.parallel.setMaxThreads(4); // programmatic twin of FUCINA_MAX_THREADS
    const n = fucina.parallel.cpuThreadCount(fucina.parallel.vector_max_threads);
    try std.testing.expect(n >= 1 and n <= 4);
    try std.testing.expect(n <= fucina.parallel.vector_max_threads);
}
```

## 2.3 Build steps (`build.zig`)

`zig build` with no step runs the default **install** step: every installed
executable lands in `zig-out/bin/` (named `fucina-<name>`). Bench and check
executables are *not* installed; they build on demand when their step runs.
Every example-runner step depends only on its own executable's
install-artifact step, so `zig build qwen3` builds just that executable;
among the `bench*` steps only `bench-gate` depends on the full install
step. Arguments after `--` are forwarded to the launched program, and
`zig build --help` lists every step and project option.

One home per command set:

- **Verification gates** (`test`, `test-fucina`, `test-models`, `test-serving`,
  `arch-check`, `doc-check`, `snippet-check`, `x86dot-check`, `cuda-check`,
  `metal-check`, `bench-check`, `bench-gate`): the gate matrix in
  [DEVELOPMENT.md §4.2](../DEVELOPMENT.md#42-run-the-gates-that-your-change-can-affect),
  with the test-root layout, snippet contract, and CI matrix in
  [DEVELOPMENT.md §7](../DEVELOPMENT.md#7-test-layout-doc-snippets-and-ci)
  (see also [§2.7](#27-test-organization-src-examples)).
- **Model runners and example applications**:
  [RUNNING-MODELS.md](../RUNNING-MODELS.md) maps every step to its
  per-example README (`examples/<name>/README.md`), which owns that
  runner's flags; [§14](14-model-families-and-example-applications.md)
  documents the per-family APIs.
- **Microbenchmarks** (`bench*`): the command list in `AGENTS.md`; protocol
  and thermal discipline in [BENCHMARK.md](../BENCHMARK.md).

## 2.4 Module graph and options wiring (`build.zig`)

`build.zig` registers three library modules and two internal microbench roots
with `b.addModule`; executables get private root modules via
`b.createModule` and pull the libraries in with `addImport`.

- **`fucina`** — root `src/fucina.zig`. The public facade: tensors, autograd,
  `ExecContext`, optimizers, ES, LoRA, GGUF/safetensors I/O ([§3](03-tensors-types-construction-and-data-access.md)–[§12](12-model-io-gguf-and-safetensors.md)). It is
  the only one of the *library* modules that receives the option set:
  `module.addOptions("build_options", options)` (the microbench roots below
  and the test-root module instances receive the same `options` object).
- **`fucina_models`** — root `src/models.zig`. The LLM/ASR stack ([§13](13-the-model-stack-fucina_models.md)). It does
  *not* get `build_options`; every module built from `src/models.zig` instead
  receives a single-key `models_build_options` module (`llguidance: bool`, read
  by `src/models/text/llguidance.zig`). It reaches the configured core exclusively
  through `models_module.addImport("fucina", module)` and the `fucina.internal`
  seam, so there is exactly one copy of the backend/exec types.
- **`fucina_serving`** — root `src/serving.zig`. The model-free HTTP serving
  transport ([§13.13](13-the-model-stack-fucina_models.md#1313-serving-srcmodelstextserving)): no options module of its own; it imports `fucina`
  and `fucina_models` (the serving contract and chat types) and nothing
  else.
- **`bench_raw`** — root `src/bench_raw.zig`, same options. Internal raw
  tensor surface (`RawTensor`, `ExecContext`, `optim`) for
  `bench/{mlp,optim,ce,conv,scatter,backward_diamond,attention_backward,train_step,facade,einsum}.zig`.
  Not part of the public facade — the root export guard in `src/fucina.zig`
  makes `fucina.RawTensor` a compile error.
- **`raw_backend`** — root `src/backend.zig`, same options. Direct kernel
  access for `bench/{backend,f16gemm,gemm,packed_gemm,gpu_dispatch,gpu_formats,q5kmoe,q8gemv,ternary}.zig`
  (`bench/membw.zig` imports neither module — the bandwidth probe is
  standalone). The
  `bench-backend` executable additionally receives a second options module
  named `bench_options` (`native_blas_kind: BlasKind`,
  `native_uses_blas: bool`, `native_blas_threads: u32`) so it can label its
  output with the native backend's BLAS configuration.

The `build_options` module is built with `b.addOptions()` and exactly these
keys (`options.addOption(T, name, value)`):

| Key | Type | Value |
| --- | --- | --- |
| `backend_kind` | `enum { scalar, native, cpu }` | `-Dbackend` |
| `blas_kind` | `enum { none, accelerate, openblas, mkl, blis, nvpl, blas }` | resolved `-Dblas` |
| `use_blas` | `bool` | `blas_kind != .none` |
| `blas_threads` | `u32` | `-Dblas-threads` |
| `max_threads` | `usize` | `-Dmax-threads` |
| `use_gpu` | `bool` | `gpu_kind != .none` |
| `gpu_kind` | `enum { none, metal, cuda }` | `-Dgpu` |
| `vector_scan` | `bool` | `-Dvector-scan` |

Only eight files outside tests import it, all inside the `fucina` module:
`src/parallel.zig`, `src/backend.zig`, `src/backend/native.zig`,
`src/backend/gpu.zig`, `src/backend/metal.zig`, `src/backend/cuda.zig`,
`src/exec/reduce.zig`, `src/exec/matmul.zig` (a `src/ag/tensor_tests/scan.zig`
test also branches on `vector_scan`, and a `src/exec/matmul_tests.zig` test skips
on `use_gpu`). The parakeet executable and
its test root get their *own* single-key `build_options`
(`parakeet_mic: bool`) — the name collides deliberately; the example reads
its key, the library module keeps its full set.

Example and bench targets are declared through two spec-driven helpers —
`addExample(b, ctx, spec)` (exe + module imports + BLAS/GPU config +
install + a run step forwarding `-- args`) and `addBench(b, ctx,
bench_check_step, spec)` (no install; registers into `bench-check`) —
with per-target special wiring (extra imports, libc, llguidance, option
modules) attached to the returned artifacts at the call site. Linking
itself is centralized in six helpers applied per executable:

- `configureBlas(step, blas_kind)` — per provider: link libc plus
  `Accelerate` (framework), `openblas`, `mkl_rt`, `blis`, `nvpl_blas`, or
  generic `blas`, with Homebrew/oneAPI/HPC-SDK library search paths *and*
  rpaths added (`/opt/homebrew/opt/{openblas,blis}`,
  `/usr/local/opt/{openblas,blis}`, `/opt/intel/oneapi/mkl/latest`,
  `/opt/nvidia/hpc_sdk`).
- `configureGpu(b, step, gpu_kind)` — `metal`: link libc + `Metal` +
  `Foundation` and compile `src/backend/metal/shim.m` (`-fobjc-arc`);
  `cuda`: link libc only (the provider `dlopen`s `libcuda.so.1`/cuBLAS via
  `std.DynLib` at runtime).
- `configureLlguidance(step, dep)` — no-op unless `-Dllguidance`; then links
  the cargo-built staticlib plus libc, and on non-macOS targets Zig's
  bundled LLVM libunwind via `link_libcpp` (the Rust FFI converts panics to
  error strings with `catch_unwind`, and glibc does not export
  `_Unwind_*`; macOS's libSystem ships an unwinder).
- `configureNamAudio` / `configureOmnivoiceAudio` / `configureParakeetAudio`
  — the vendored miniaudio C shims (`apps/nam/audio_shim.c`, plus
  `midi_shim.c` for NAM and `apps/omnivoice/play_shim.c` for playback),
  with CoreAudio/CoreMIDI frameworks on macOS; elsewhere miniaudio `dlopen`s
  its backend through libc.

## 2.5 Consuming Fucina from another project

Fucina is an ordinary Zig package: `build.zig.zon` names it `.fucina`, the
repository is tagged (`v0.3.0`), and the three library modules are exported by
`build.zig` (`b.addModule`), so the standard path is the package manager.
From the consumer project:

```sh
zig fetch --save git+https://github.com/matteo-grella/fucina#v0.3.0
```

```zig
// build.zig (consumer) — verified against Zig 0.16.0
const fucina_dep = b.dependency("fucina", .{
    .target = target,
    .optimize = optimize,
    // Any §2.2 build option passes through by name, e.g.:
    //   .blas = .none, .backend = .native, .@"max-threads" = @as(usize, 4),
});
exe.root_module.addImport("fucina", fucina_dep.module("fucina"));
exe.root_module.addImport("fucina_models", fucina_dep.module("fucina_models")); // optional
exe.root_module.addImport("fucina_serving", fucina_dep.module("fucina_serving")); // optional
```

`@import("fucina")` / `@import("fucina_models")` / `@import("fucina_serving")`
then work exactly as in every snippet of this reference; omit the
`fucina_models` and `fucina_serving` imports for tensor/training-only
consumers. In dependency builds the exported modules
carry their own BLAS/GPU link inputs (link inputs propagate through module
imports), so the default macOS configuration links Accelerate with no
extra consumer steps and `-Dgpu=metal` brings its shim along — no
`configureBlas`/`configureGpu` replication. Option defaults match the
in-tree build ([§2.2](02-toolchain-build-and-project-wiring.md#22-build-options-buildzig)); pass `.blas = .none` for a zero-system-dependency
build. Two limits: `.llguidance = true` is not supported through the
package manager (the vendored cargo build is designed for an in-tree
checkout — vendor the repo for constrained decoding), and the API is
pre-1.0 ([§1.5](01-introduction-and-mental-model.md#15-stability)) — pin the tag or a commit (`#<sha>`) and expect churn
between tags.

**Vendoring fallback.** A consumer can instead vendor the repository
(git submodule, subtree, or plain copy) and wire the modules in its own
`build.zig` with the same `std.Build` calls the in-tree build uses. The
option enums must be re-declared, but only the *field names* matter — the
fucina sources switch on them by name — and every key is required
(compilation of `src/parallel.zig`/`src/backend.zig`/`src/backend/gpu.zig`/`src/exec/reduce.zig`
fails on a missing key). Keep the two derived booleans consistent with
their enums.

```sh
git submodule add https://github.com/matteo-grella/fucina vendor/fucina
```

```zig
// build.zig (consumer) — verified against Zig 0.16.0
const std = @import("std");

const BackendKind = enum { scalar, native, cpu };
const BlasKind = enum { none, accelerate, openblas, mkl, blis, nvpl, blas };
const GpuKind = enum { none, metal, cuda };

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The comptime configuration the fucina sources read as
    // `@import("build_options")`. Every key is required.
    const options = b.addOptions();
    options.addOption(BackendKind, "backend_kind", .native);
    options.addOption(BlasKind, "blas_kind", .none);
    options.addOption(bool, "use_blas", false); // keep == (blas_kind != .none)
    options.addOption(u32, "blas_threads", 0);
    options.addOption(usize, "max_threads", 8);
    options.addOption(bool, "use_gpu", false); // keep == (gpu_kind != .none)
    options.addOption(GpuKind, "gpu_kind", .none);
    options.addOption(bool, "vector_scan", false);

    const fucina = b.addModule("fucina", .{
        .root_source_file = b.path("vendor/fucina/src/fucina.zig"),
        .target = target,
        .optimize = optimize,
    });
    fucina.addOptions("build_options", options);

    const fucina_models = b.addModule("fucina_models", .{
        .root_source_file = b.path("vendor/fucina/src/models.zig"),
        .target = target,
        .optimize = optimize,
    });
    fucina_models.addImport("fucina", fucina);
    // fucina_models's own comptime configuration, read as
    // `@import("models_build_options")` — required by every module built from
    // `src/models.zig` (src/models/text/llguidance.zig reads the boolean `llguidance`
    // key; false keeps the engine stubbed). `true` additionally needs the
    // cargo staticlib build + link from fucina's build.zig (§2.2
    // `-Dllguidance`).
    const models_options = b.addOptions();
    models_options.addOption(bool, "llguidance", false);
    fucina_models.addOptions("models_build_options", models_options);

    const exe = b.addExecutable(.{
        .name = "myapp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.root_module.addImport("fucina", fucina);
    exe.root_module.addImport("fucina_models", fucina_models);
    // Non-default -Dblas / -Dgpu configurations also need the link steps
    // from fucina's build.zig (configureBlas / configureGpu): frameworks,
    // system libraries, and the Metal shim C source.
    b.installArtifact(exe);
}
```

The application code then imports the modules by the names given to
`addImport`: `const fucina = @import("fucina");` and
`const models = @import("fucina_models");`. `fucina_models` is optional — omit it
(and its `addImport`) for tensor/training-only consumers. For a BLAS or GPU
configuration, replicate the corresponding `configureBlas`/`configureGpu`
body from the in-tree `build.zig` on the consumer executable (the Metal shim
path becomes `vendor/fucina/src/backend/metal/shim.m`). The public API is
not yet stable (`README.md` says so explicitly); pin the vendored commit.

## 2.6 Runtime environment variables

Every knob in the core-runtime and GPU tables is read **once**, at the
first `tuning.get()`, and cached; changing the process environment
afterwards has no effect. The example/test gates in the last table are
plain `getenv` calls re-read at every use.
Numeric knobs that fail to parse fall back to their defaults;
`FUCINA_MAX_THREADS`-style positive-integer knobs ignore unset/invalid/zero
values. On Linux without libc the lookup scans `/proc/self/environ`
(`src/parallel.zig`), so every variable also works in static builds.

With four exceptions (`FUCINA_MAX_THREADS`, a bootstrap read in
`src/parallel.zig`; the string-valued `FUCINA_GPU_KERNELS`; the profiling
flags `FUCINA_POOL_PROFILE`/`FUCINA_MM_PROFILE`), every variable
below is a leaf of one typed table, `fucina.tuning.Table`
(`src/tuning.zig`), and its name derives from its field path: `FUCINA_` plus
the path segments upper-cased and joined with `_` (`decode_compact` reads
`FUCINA_DECODE_COMPACT`, `gpu.min_work.attn` reads
`FUCINA_GPU_MIN_WORK_ATTN`; a leaf named `base` or `enabled` names its
group, so `gpu.min_work.base` reads `FUCINA_GPU_MIN_WORK` and `gpu.enabled`
reads `FUCINA_GPU`). Booleans: a set, non-empty value whose first character
is `0` forces the route off, any other set, non-empty value forces it on,
unset keeps the measured default: one spelling per gate, `FUCINA_X=0`
where the old `FUCINA_NO_X=1` used to be. Integers parse base-10; the
top-level crossovers treat `0` and garbage as unset, while leaves under
`gpu` accept `0` as a meaningful value (a zero work floor always offloads).
The whole table is read once at the first `tuning.get()` and cached;
`tuning.setField`/`tuning.set` pin fields programmatically (the per-gate
`set*` test hooks forward there, and pinning `null` re-arms the env/default
value). Policy that can differ per workload goes through
`fucina.tuning.Overrides`, the same field tree with every leaf optional:
`ExecContext.setTuning(.{ ... })` overrides any table field for that
context only, consulted by the routes that support per-context policy
(first consumer: the CPU f32 weight-shadow route — `cpu_f32_shadow` /
`cpu_f32_shadow_min_m` — so two contexts in one process can run different
shadow policy).

**Core runtime** (`src/parallel.zig`, `src/exec/conv.zig`,
`src/exec/attention.zig`, `src/exec/matmul.zig`, `src/ptqtp_gguf.zig`,
`src/store/expert_store.zig`):

| Variable | Effect | Default |
| --- | --- | --- |
| `FUCINA_MAX_THREADS` | Lowers the worker count below the `-Dmax-threads` ceiling (mirrors llama.cpp `-t`). Never raises it. Consulted on the first `cpuThreadCount` call; a prior `setMaxThreads` wins. | unset (detected CPU count — clamped to physical cores on SMT hosts and to performance cores on Apple Silicon — capped by the ceiling) |
| `FUCINA_SPIN_BUDGET` | Overrides the worker-team spin-then-park window (`src/thread.zig` BarrierPool; `0` = park immediately is a valid override, values above `u32` are ignored). Consulted at pool init through the tuning table. Workload-coupled; the default is deliberate. | unset (32768 spins; `0` when the team exceeds the physical-core count — spinning while oversubscribed starves the descheduled participants) |
| `FUCINA_POOL_PROFILE=1` | Emits one `[pool-trace]` line per `BarrierPool` dispatch with span, claim chunk, and each participant's first-claim/completion offsets and task count. Diagnostic only; read once when the team is created. | off |
| `FUCINA_WINOGRAD=1/0` | Forces the Winograd conv2d route on/off (A/B + emergency revert switch). | on for no-BLAS builds, off when a platform BLAS backs the matmul |
| `FUCINA_WINOGRAD_F4=0` | Pins Winograd-routed large maps to the F(2×2,3×3) tier. | F4 tier enabled |
| `FUCINA_WINOGRAD_F4_MIN` | Minimum output spatial size for the F4 tier. | `14` |
| `FUCINA_WINOGRAD_F4_MAXCIN` | Maximum input channels for the F4 tier (deep-channel maps run faster on F2). | `56` |
| `FUCINA_CONV_BWD_GEMM=0` | Pins the `groups == 1` conv2d backward entries to the direct gather kernels instead of the GEMM (matmul + im2col/col2im) decomposition (A/B + emergency revert switch). | GEMM route on |
| `FUCINA_ATTN_BWD_STATS=1/0` | Forces the forward-saved-stats route of the attention-backward softmax reconstruction (`src/exec/attention.zig`) on/off (A/B + emergency revert switch) — the two routes agree to f32 roundoff, not bitwise; only consulted when the autograd record saved forward stats (the stats-less exec path always recomputes). | on |
| `FUCINA_ATTN_BWD_BLAS=0` | Reverts the attention-backward contraction tiles from the BLAS-strip route (the per-tile contractions issued as strided sgemm strips) to the register-tiled route (`src/exec/attention.zig`; A/B + escape hatch for parity work) — the two routes agree to f32 roundoff, not bitwise. Only consulted on BLAS-backed native builds; elsewhere the register-tiled route always runs. | BLAS-strip route on (BLAS builds) |
| `FUCINA_CPU_F32_SHADOW=1` | Opt-in (`src/exec/matmul.zig`): attaches a widen-once f32 shadow to a 16-bit weight's storage and routes m ≥ 32 GEMMs through the BLAS f32 path (decode stays on the streaming kernels). +4 bytes/weight resident; leave off when training 16-bit weights in place. CPU builds only. | off |
| `FUCINA_CPU_F32_SHADOW_MIN_M` | Overrides the shadow route's m ≥ 32 crossover. | `32` |
| `FUCINA_PTQTP_FOLD=0` | Serves tie-fitted PTQTP MoE plane sets through the per-plane path instead of the folded one-pass form (`src/ptqtp_gguf.zig`; A/B on one binary; a striped expert-store L2 tier only covers unfolded layers, so this trades the halved cache-hit dot for L2 coverage). | folded serving on |
| `FUCINA_MOE_LRU=1` | Forces the pure-LRU victim scan in the MoE expert store (`src/store/expert_store.zig`; A/B on one binary). | heat-aware eviction |
| `FUCINA_MOE_L2_CACHED=1` | Keeps the expert-store L2 tier page-cached instead of uncached I/O (`src/store/expert_store.zig`). | uncached |

**GPU offload** (read by both providers unless noted;
`src/backend/metal.zig`, `src/backend/cuda.zig`; see [§9](09-backends-cpu-simd-blas-threading-and-gpu-offload.md)):

| Variable | Effect | Default |
| --- | --- | --- |
| `FUCINA_GPU` | Kill switch: a value starting with `0` disables the GPU provider entirely. | enabled on `-Dgpu` builds |
| `FUCINA_GPU_MIN_WORK` | Base f32 GEMM offload gate, in m·n·k work units. | Metal `2^32` (cold single-op crossover); CUDA `2^30` (the transient floor below still dominates ordinary host RHS) |
| `FUCINA_GPU_MIN_WORK_F16` | f16 GEMM gate. | `2^27` (lower — the CPU f16 competitor has no AMX-class arm) |
| `FUCINA_GPU_MIN_WORK_F16_RESIDENT` (cuda) | f16 GEMM/GEMV gate when the RHS already has a device address; permits small-m decode without admitting a streamed weight. | `2^20` |
| `FUCINA_GPU_MIN_WORK_16BIT_RESIDENT` (metal) | f16/bf16 GEMM gate when the RHS is already Metal-mapped; admits batched decode at m ≥ 16 and the lm-head row while narrow decode stays on the CPU streaming kernels. | `2^27` |
| `FUCINA_GPU_MIN_WORK_GEMV` | Resident dense-f32 GEMV/small-m GEMM gate (`m <= 8`; nonresident CUDA RHS is refused). | `2^24` |
| `FUCINA_GPU_MIN_WORK_RESIDENT` (cuda) | Dense-f32 GEMM/batched-GEMM gate when the RHS already has a device address. | `2^27` (512³; 256³ loses to OpenBLAS-32 on the reference host) |
| `FUCINA_GPU_MIN_WORK_QMOE` | Grouped quantized MoE GEMM gate; setting it also re-seeds the dense-Q6 gate. | `2^30` |
| `FUCINA_GPU_MIN_WORK_DENSE_Q4` | Dense Q4_K model-weight gate against the load-time-packed CPU fallback. | Metal `2^30`; CUDA `2^27` |
| `FUCINA_GPU_MIN_WORK_DENSE_Q5` (cuda) | Dense Q5_K model-weight gate against the load-time-packed CPU fallback. | `2^24` |
| `FUCINA_GPU_MIN_WORK_DENSE_Q6` | Dense Q6_K gate; overrides both the compact/raw and packed-CPU tiers. | compact/raw `2^22`; packed Metal `2^31`, CUDA `2^24` |
| `FUCINA_GPU_MIN_WORK_DENSE_Q8` | Dense Q8_0 model-weight gate against the load-time-packed CPU fallback. | Metal `2^29`; CUDA `2^24` |
| `FUCINA_GPU_MIN_WORK_DENSE_TQ2` (metal) | Dense/PTQTP ternary TQ2_0 gate against the x4 interleaved CPU kernels. | `2^25` |
| `FUCINA_GPU_QMOE_MIN_FILL` | Tile-occupancy gate (percent) for grouped MoE: small expert batches whose 32-row tiles would run mostly empty stay on CPU; `0` disables the gate, `>100` never passes it. | `50` |
| `FUCINA_GPU_TRACE` | Non-`0` first character enables dispatch tracing; dump via `fucina.internal.gpu.traceDump()` (no-op when off). | off |
| `FUCINA_GPU_TF32` (cuda) | Non-`0` opts f32 GEMMs into TF32 tensor cores (default is strict FP32). | off |
| `FUCINA_GPU_MIN_WORK_TRANSIENT` (cuda) | Work floor for *non-resident* operands (each crossing PCIe per call); an `m ≥ 128` row floor applies alongside it. | `2^33` |
| `FUCINA_GPU_MIN_WORK_ATTN` | Attention work floor, in q·kv·heads·d units, for the exec-tier grouped attention forward (f32 and f16 KV, softmax stats for training); on CUDA the same floor also gates the runner's fused prefill seam over the same kernel. | Metal `2^29`; CUDA `2^28` |
| `FUCINA_GPU_DECODE` (cuda) | Non-`0` enables opt-in quantized decode for m ≤ 8 and resident weights only (GEMV generally; Q5_K uses tiled MMA at m=4..8). | off |
| `FUCINA_GPU_MIN_WORK_DECODE_Q5` (cuda) | Q5_K-only decode work gate after `FUCINA_GPU_DECODE=1`; rejects the compact CPU kernel's measured 1×4096² win. | `3·2^23` |
| `FUCINA_GPU_QUANT_MMA` (cuda) | A value starting with `0` disables the tensor-core Q4_K/Q5_K/Q6_K/Q8_0 kernels and selects the scalar-FFMA fallback (diagnostic A/B switch). | enabled on compute capability ≥ 7 |
| `FUCINA_GPU_QUANT_SPLIT_K` (cuda) | A value starting with `0` disables the on-stream split-K/reduction used to fill idle SMs for underfilled dense quantized prefill (diagnostic A/B switch). | enabled when the N64 output grid fills less than roughly 7/8 of the SMs |
| `FUCINA_GPU_VRAM_BUDGET` (cuda) | Weight-residency budget in bytes; `0` disables the bound. | 80% of free VRAM at init |
| `FUCINA_GPU_KERNELS=src` (cuda) | NVRTC-recompiles the vendored kernels from `kernels.cu` instead of loading the committed PTX (dev loop; `tools/gen_cuda_ptx.sh` regenerates the PTX). String-valued, so it lives outside the tuning table as a direct env read. | committed PTX |

**Model I/O + LLM stack** (`src/weights.zig` [§13.2](13-the-model-stack-fucina_models.md#132-weight-loading-srcweightszig), `src/models/qwen3/train.zig`,
`src/models/inkling/mmproj.zig`; read once and cached like the tables above):

| Variable | Effect | Default |
| --- | --- | --- |
| `FUCINA_NORM_QUANT_FUSED=1/0` | Forces the fused normalize+quantize+packed-GEMM route of `linearSeqNormed` on/off (prefill shapes on the packed CPU arms only; the fused route matches the unfused `rmsNormMul` + linear pair to f32 roundoff, not bitwise). | on |
| `FUCINA_DECODE_COMPACT=1/0` | Routes decode-shape (m < 4) no-grad K-quant matmuls through the GGUF-native compact blocks instead of the byte-expanded packed layout — bitwise-equal, fewer weight bytes streamed (Q4_K ~1.92×, Q5_K ~1.57×, Q6_K 1.30×). | on |
| `FUCINA_FUSED_DISTILL=0` | Forces the composed logits + `cartridge.distillLoss` tail instead of the fused distill route in cartridge training (`src/models/qwen3/train.zig`; A/B + emergency revert — the fused route matches it to f32 roundoff, not bitwise). | fused route on |
| `FUCINA_MM_PROFILE=1` | Per-stage timing profile of the Inkling multimodal-projector encode (`src/models/inkling/mmproj.zig`; read once at load). | off |

**Examples and test gates** (`examples/`):

| Variable | Effect | Default |
| --- | --- | --- |
| `FUCINA_NAM_PROFILES` | Profile directory for the `nam` CLI (`--profiles-dir` overrides it). | `nam-profiles` |
| `OMNIVOICE_PARITY=1` | Enables the OmniVoice parity suites under `zig build test` (need model files under `models/omnivoice/` and locally captured reference goldens); unset, they `error.SkipZigTest`. | skipped |
| `OMNIVOICE_AUDIO_DEVICE_TESTS=1` | Enables the speaker-playback device tests. | skipped |
| `OMNIVOICE_TOKENIZER_GGUF=<path>` | Points the real-codec-GGUF load test at a tokenizer GGUF. | skipped |
| `NANOCHAT_PARITY=1` | Enables the nanochat parity suites under `zig build test` (need locally captured reference goldens); unset, they `error.SkipZigTest`. | skipped |
| `FUCINA_TEST_VERBOSE` | Any value re-enables the facedetect/nanochat per-case test-progress prints on stderr (`examples/{facedetect,nanochat}/testlog.zig`); failure-path prints stay on regardless. | silent |
| `FUCINA_TEST_REQUIRE_MODELS` | Any value turns a missing model/fixture in a model-gated test into a FAILURE instead of a skip (`src/models/test_support.zig`) — the rig-run guard against silent skips. Needs libc for the env read; libc-free builds treat it as unset. | skip |

## 2.7 Test organization (`src/`, `examples/`)

Tests live in sibling `*_tests.zig` files forwarded from their production
files, `zig build test` runs one test root per library module and
test-carrying example, and every runnable fenced `zig` block in these
chapters is itself a test (`zig build snippet-check`). The layout, the skip
discipline
for asset- and feature-gated suites, and the snippet authoring contract
live in
[DEVELOPMENT.md §7](../DEVELOPMENT.md#7-test-layout-doc-snippets-and-ci).

## 2.8 Continuous integration (`.github/workflows/ci.yml`)

CI runs `zig build test` plus every gate above on an ubuntu + macos
matrix, with scalar, no-BLAS, and llguidance legs; the step list lives in
[DEVELOPMENT.md §7.3](../DEVELOPMENT.md#73-continuous-integration-githubworkflowsciyml)
and in `.github/workflows/ci.yml` itself.
