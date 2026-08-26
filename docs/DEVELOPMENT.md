<!-- docs-nav: group="Project" title="Development" weight=52 -->
# DEVELOPMENT — how to build on Fucina

The working method for new development on this codebase — features, kernels,
model families, training machinery — written for the next contributor,
human- or agent-driven. `AGENTS.md` is the entry point (toolchain, commands,
repo map, house rules); [PORTING.md](PORTING.md) is the method for ports
specifically; the [reference](reference/00-index.md) is the API contract. This file is
the connective tissue: the invariants a change must respect, what already
exists so you don't rebuild it, which existing code to start from, and the
delivery loop that takes a change from idea to merged.

The one-line version: **find the existing capability first, extend it at its
designed seam, verify against the reference backend and the gates, measure
before claiming speed, and report what you actually did.**

## 1. Invariants

Each invariant states the rule, where it is enforced, and what a violation
looks like. The ones marked **review-only** have no mechanical gate — nothing
fails automatically when you break them, which is exactly why they are
listed here.

### 1.1 Layering is a one-way street

A band may depend only on bands at or below it: apps (`examples/`, `apps/`,
`tools/`, `bench/`) → models (`src/models/`) → facade (`src/fucina.zig`) → autograd/training
(`src/ag/`, optim/es/lora/persistence) → tagged ops (`src/tag_ops.zig`) →
exec runtime (`src/exec/`) → backends (`src/backend/`) → tensor/storage/dtype.
`fucina_models` files import the `fucina` *module* (public surface plus
`fucina.internal`), never individual `src/*.zig` files.

*Enforced by:* `zig build arch-check` — the production import graph of `src/`,
`examples/`, `apps/`, `bench/`, and `tools/` must have zero strongly-connected
components AND zero band inversions (AST-based, test-aware). Band direction is checked against the layer table in
[ARCHITECTURE.md](ARCHITECTURE.md), encoded as `band_table` in
`tools/check_import_graph.zig`: production layer inversions are a failed
build, full stop. A production file in no band fails the check as well, so a
new `src/` root cannot slip in unclassified.

*Violation:* exec importing ag; an example importing `src/tensor.zig`
directly; family-specific logic inside `src/exec/` (see §1.8).

### 1.2 Eager and local — no graph compiler (review-only)

No global graph object, no fusion pass, no lazy evaluation, no planner. Every
op validates, allocates through `ExecContext`, and runs a kernel immediately.
This is a deliberate design stance, not debt. An app-level compiled replay is
fine *inside an example* (`apps/facedetect/graph.zig` is the precedent);
a core IR is not. Don't add one without a concrete design.

### 1.3 One public tensor, comptime tags, sealed raw layer

`fucina.Tensor(spec)` is the only public tensor. Tags and rank are comptime;
sizes are runtime. What each dtype branch can do is enforced by the type
system: `.f32` differentiates; f16/bf16 add forward math and 16-bit-leaf
autograd (gradients are always f32); integers/bool are constants;
block-quantized tensors dequantize, gather, and serve as matmul RHS — nothing
else. The raw tensor is deliberately not exported (a comptime guard in
`src/fucina.zig` makes it a compile error); in-tree internal access goes
through `fucina.internal`.

*Violation:* a second public tensor type; runtime tag values; loosening a
dtype branch with a runtime check where a compile error is the design.

### 1.4 Explicit ownership and the deinit convention

Storage is refcounted and owned; slices and views borrow. Every op returns an
owned result; `defer x.deinit()` is the norm, and that deinit is what drives
buffer recycling through the `BufferPool`. Training uses exec scopes
(`openExecScope`/`closeExecScope`) to own intermediates implicitly; pure
inference stays deinit-as-you-go. `deinit(self)` when the struct carries its
own ctx/allocator, `deinit(self, allocator)` for POD-ish holders; either way
end with `self.* = undefined`. An arena for transients was evaluated and
**rejected** — [MEMORY-MODEL.md](MEMORY-MODEL.md) records why (peak memory,
cache locality, refcounted views, training lifetimes); do not reintroduce
one.

*Enforced by:* the testing allocator (leaks fail tests), the BufferPool's
`outstanding == 0` teardown assert, and the `undefined` tripwire in Debug.

### 1.5 Kernels never allocate — outputs are exec-supplied (review-only)

Backend compute leaves (`backend/vector/`, `backend/quant/` dots) are
allocation-free and infallible; results go into buffers the
`ExecContext` supplies. The one sanctioned exception is the
quantized-RHS *dispatch* tier (`matmul2DQuantizedRhs*`), which takes an
explicit allocator for per-call LHS-quantization scratch. Don't add
allocation below that tier.

### 1.6 Validate, then call an unchecked kernel (review-only)

Shape/stride/alignment/contiguity checks live in the caller/runtime; the
kernel underneath is small, unchecked, and fast. ReleaseFast drops safety
checks, so a kernel that only behaves because Debug catches it is broken —
prove invariants, don't use checks as logic.

### 1.7 Comptime dispatch, exhaustive switches

Backend (`-Dbackend`), BLAS provider, and GPU provider are chosen at build
time; dead arms are never analyzed. Prefer exhaustive `switch` over
dtype/backend so adding a variant *forces* edits everywhere — the compile
error is the enforcement. A silent `else` that swallows a new dtype defeats
the design. Related trap: a bare `-Dtarget` cross-build drops to the
architecture baseline and silently loses the fast kernel arms — pin `-Dcpu`
or build on the machine that runs it.

### 1.8 Placement policy

Reusable engines other `src/models` consumers should import go in
`src/models/<family>/`; single-purpose parity ports and their DSP/IO plumbing
stay app-local in `apps/<name>/`; generic helpers stay flat in
`src/models/`; family-specific kernel orchestration lives in the family over
the `fucina.internal` seam, never inside the generic exec runtime; shared
cross-family scheduling lives once in exec and is re-exported
(`exec/moe_chain.zig` is the pattern). New core tensor ops belong in the
exec/backend bands — but a good port usually needs none: nanochat is
entirely app-local over the public facade. The apps band itself has two
homes: `examples/<name>/` is teaching code (one `main.zig`, a README is
fine, no test files, no C shims, no vendored assets; it exists to be read
and copied); `apps/<name>/` is everything with product or port shape
(multi-file, own tests, shims, goldens, or a real CLI surface).

### 1.9 Determinism is a contract

`src/rng.zig` is the repo-owned deterministic RNG; its (seed → values)
mapping is a checkpoint contract (dropout masks, APOLLO projections, ES
noise are regenerated from seeds, not serialized). The SFT loader's
`(seed, epoch) → permutation` mapping is golden-pinned. Parallel kernels are
bitwise-deterministic for any thread count, or they document their rounding
class precisely (the `-Dvector-scan` option text is the model to imitate).
Never swap in `std.Random` for anything seed-persisted; never change the fill
algorithms without accepting that existing checkpoints break.

### 1.10 Scalar is the specification

The scalar backend (`-Dbackend=scalar`) is the executable reference: native
and scalar must agree, and `src/backend/parity_test.zig` holds them together.
The scalar leg runs before merge when the change touches `src/backend/` or
`src/exec/` (once, on the final code — it is slow by design); other changes
skip it, since `parity_test.zig` already diffs both backends inside every
native `zig build test`. Everything integer is bit-exact across
architectures; float tile kernels document association-order tolerance
instead.

## 2. Check before you build

The most common failure mode of a capable contributor on this codebase is
rebuilding something that exists. Search this table first; the § pointers go
into the [reference](reference/00-index.md), whose snippets are machine-verified.

| You need… | It exists as… | Where |
| --- | --- | --- |
| A contraction (matmul, batched, multi-index) | `einsum` — THE contraction engine; `dot` is its special case. Don't hand-roll permute+matmul chains. | [§4.8](reference/04-tensor-operations.md#48-dot-tag-directed-contraction-srcagtensorzig-srctag_opszig)–4.9 |
| A new pointwise op with autograd | `elementalUnary`/`elementalBinary` — supply scalar fwd/bwd fns, get a SIMD-chunked parallel op with a VJP. | [§4.4](reference/04-tensor-operations.md#44-unary-ops-srcagtensorzig-srcbackendopszig) |
| Indexing / slicing / functional updates | `select`, `slice`, `sliceStep`, `indexSelect`, `gather`, `setSlice`, `setRows`, `indexAdd`, `scatterAdd`, `maskedScatter`, `where`, `oneHot` | [§3.7](reference/03-tensors-types-construction-and-data-access.md#37-views-and-structural-ops-srcagtensorzig-srctag_opszig-srcexecgather_scatterzig), [§4.17](reference/04-tensor-operations.md#417-indexing-assembly-and-functional-updates-srcagtensorzig) |
| Reductions, scans | SIMD-promoted sum/mean/max/min/prod/logsumexp; `cumsum`/`cumprod` (+ `-Dvector-scan`) | [§4.7](reference/04-tensor-operations.md#47-reductions-and-scans-srcagtensorzig) |
| Attention | `groupedAttention`: causal/bidirectional, sliding window, additive bias, sinks, ALiBi-style `max_bias`, f32/f16/q8 KV, saved-stats backward | [§4.13](reference/04-tensor-operations.md#413-attention-srcagtensorzig) |
| RoPE | interleaved/half/tail-aligned modes, partial rotary, freq-factor (YaRN-style) tables, inverse tables, hand-fillable `RopeTable` | [§4.12](reference/04-tensor-operations.md#412-rotary-position-embedding-srcagtensorzig-srcexecropezig) |
| Norms, softmax, losses | rmsNorm family (incl. fused mul/add/rope), layerNorm, groupNorm; softmax with scale/mask/sinks; `crossEntropy` (smoothing, ignore-index, accum scale), mse/huber/bce/kl/nll | [§4.10](reference/04-tensor-operations.md#410-softmax-family-srcagtensorzig-srcexecsoftmaxzig)–4.11, [§4.15](reference/04-tensor-operations.md#415-losses-and-similarity-srcagtensorzig-srcexeclosszig) |
| Vision / conv | channel-last conv2d (im2col GEMM + Winograd), conv1d/causal/transpose, pool2d, prelu, channelAffine, upsample, zeroPad2d — all with autograd | [§4.14](reference/04-tensor-operations.md#414-convolution-and-channel-last-vision-ops-srcagtensorzig) |
| MoE | `routerTopK`, `moeExpertFfn`/`Batch`, `MoeRhs` packed containers, `moe_chain` scheduling, disk-streamed experts (`ExpertStore`) for models larger than RAM | [§4.16](reference/04-tensor-operations.md#416-selection-argmax-topk-sort-routertopk-srcagtensorzig-srcexectopkzig), [§4.18](reference/04-tensor-operations.md#418-moe-facade-entries-srcexecmoezig-srcexeczig), [§13.2](reference/13-the-model-stack-fucina_models.md#132-weight-loading-srcweightszig) |
| Autograd machinery | seeded backward (`backwardWithGrad`), `noGrad`, activation checkpointing, `customVjp`, `gradcheck`; VJP inventory in [§5.8](reference/05-automatic-differentiation.md#58-vjp-coverage-inventory-srcagbackward) | [§5](reference/05-automatic-differentiation.md) |
| Training | SGD/AdamW/Adam/Muon/APOLLO (torch-golden-parity), `OptimizerSet` param groups, LR schedules, clipping, `ParamRegistry`, LoRA adapters, ES (incl. ternary-native), 16-bit leaves with f32 grads + optimizer masters | [§11](reference/11-training-optimizers-evolution-strategies-lora-and-checkpoints.md) |
| Quantized weights | hot packed kernels (Q4_K/Q5_K/Q6_K/Q8_0/TQ2_0 + 2-bit expert tier), cold decode (IQ*, FP4…), byte-exact ggml encoders, PTQTP trit-planes, fake-quant round trips (FP8/FP4 microscaling, Hadamard, f16) | [§10](reference/10-quantization.md) |
| Persistence | GGUF read/write/transcode (byte-verbatim re-emit), safetensors, named state dicts with alias remapping, training-checkpoint directories, `export-gguf` (incl. LoRA merge) | [§12](reference/12-model-io-gguf-and-safetensors.md) |
| LLM plumbing | `LinearWeight` (dispatch to BLAS/Metal/CUDA/quant kernels is *inside* — never hand-roll a linear), `gguf_meta` readers, KV cache (f16/q8_0) + crash-safe persistence, BPE + SPM tokenizers, sampler + `LogitProcessor` + llguidance constrained decoding, generic `Conversation` chat engine (+ `sendBatch`), speculative decoding cascade + grammar-constrained drafting, native MTP | [§13](reference/13-the-model-stack-fucina_models.md) |
| Parallelism / infra | worker team + `parallelChunks` (this *is* the parallel-loop contract), `BufferPool`, `RhsLifetime` RHS caching, deterministic RNG, GPU offload gates | [§6](reference/06-the-execution-runtime-execcontext-and-the-memory-model.md), [§9](reference/09-backends-cpu-simd-blas-threading-and-gpu-offload.md) |

If a capability is genuinely missing, check the design records first —
[SPECULATIVE.md](SPECULATIVE.md), [CONSTRAINED-DECODING.md](CONSTRAINED-DECODING.md),
[TERNARY.md](TERNARY.md), [PTQTP.md](PTQTP.md), [MEMORY-MODEL.md](MEMORY-MODEL.md) —
several "obvious" additions (arenas, LUT kernels, negative-step views,
do-concurrent wrappers) were evaluated and rejected with measurements, and
the records say why.

## 3. Start from a template

Every kind of work has a best-in-class precedent in the tree. Read it before
designing.

| New work | Start from | Why |
| --- | --- | --- |
| LLM family, llama-shaped | `src/models/qwen3/` | cleanest dense+MoE model, spec decode, trainer |
| LLM family, SPM tokenizer / sliding window / MoE engines | `src/models/gemma/` | per-layer KV geometry, MoE engine with GPU arm |
| Hybrid/recurrent blocks | `src/models/qwen35/` | Gated-DeltaNet over the same loader conventions |
| MLA / MTP / streamed-expert giants | `src/models/deepseek4/` | compressed KV, hyper-connections, out-of-core experts |
| Non-autoregressive decoder | `src/models/diffusion_gemma/` | two forward modes over one weight set |
| Pure-CNN vision port | `apps/facedetect/` | load-once models, BN-fold at load, byte-identical JSON goldens |
| VLM port | `apps/locate_anything/` | ViT tower + LM, custom RopeTable, MTP box decode |
| ASR / encoder stack | `src/models/parakeet/` | the reusable-family precedent |
| TTS / codec port | `apps/omnivoice/` | codec parity, RNG parity, chunked streaming |
| Streaming DSP / effects | `apps/nam/` | streaming engines, format interchange, live IO |
| Training pipeline | `apps/nanochat/`, `examples/spirals/main.zig`, `apps/finetune/main.zig` | full pretrain→SFT→chat; minimal optimizer demo; LoRA on a real GGUF |
| HTTP/API frontend | `apps/lmserve/main.zig` | OpenAI-compatible mapping tables, SSE, backend matrix |

### 3.1 Adding one tensor op, end to end

One differentiable float op touches a fixed file list. Walk it top to
bottom; every stop is either compile-checked or named here so nothing is
left to memory. Use an existing op in the same domain as the line-level
template (e.g. `softmax` for a row op, `maxPool2d` for a pool op).

1. `src/backend/ops.zig` — only if the op needs a new selector enum
   member. The consuming switches are exhaustive, so every arm site below
   becomes a compile error until handled. One exception is named in step 8.
2. `src/backend/interface.zig` — add the kernel name to the set (and to
   `generic_names`/`pool_free_names` as applicable). `conform` then forces
   both providers.
3. `src/backend/cpu.zig` — the scalar reference implementation. This is
   the specification (see 1.10).
4. `src/backend/native.zig` + `src/backend/vector/<domain>.zig` — the
   SIMD implementation, `pc`-first signature.
5. `src/backend/parity_test.zig` or the domain's `vector/*_tests.zig` —
   pin scalar and native to each other.
6. `src/exec/<domain>.zig` — the op body over `*ExecContext` (validate,
   then call the unchecked kernel), plus one alias line in `src/exec.zig`
   under the domain's banner.
7. `src/ag/backward/<domain>.zig` — the VJP (the facade mixin of the same
   domain imports the file directly; there is no central alias table).
8. If the op is a `UnaryOp`: classify it in
   `ag/backward/elementwise.zig`'s `unaryUsesOutput` — the switch is
   exhaustive, and a wrong classification is a wrong gradient, so decide
   deliberately whether the derivative is cheaper from the forward output.
9. `src/ag/tensor/float/<domain>.zig` — the facade method (tag algebra,
   result finishing), plus one alias line in `src/ag/tensor.zig`.
10. `src/ag/tensor/typed/<math|int|widened>.zig` — only if the op should
    also exist on non-f32 constant tensors (native over the stored dtype,
    integer-only, or the f16/bf16 widened family). A view or data-movement
    op with a dtype-generic kernel goes in `src/ag/tensor/views.zig`
    instead, once, for every scalar dtype.
11. GPU offload (only if warranted by measurement): the provider seam in
    `src/backend/gpu_provider.zig` plus both providers and `gpu_none.zig`;
    the conformance check forces all three.
12. Tests: `src/exec/<domain>_tests.zig` (exec semantics),
    `src/ag/tensor_tests/<domain>.zig` (facade + gradcheck; forward the
    file from `src/ag/tensor.zig`'s test block).
13. Docs: the op's row in `docs/reference/04-tensor-operations.md` (or the
    domain chapter); runnable snippets are compile-gated by
    `zig build snippet-check`.

Gates for the whole loop: `zig build test`, `zig build test-fucina
-Dbackend=scalar`, `zig build arch-check`, and `zig build bench` (or the
domain bench) when the op is hot-path (see 4.3).

## 4. The delivery loop

### 4.1 Plan with mechanical accepts

Structure work as items with three parts — *Do* (imperative, with code
anchors), *Accept* (a mechanical gate: an exit code, a named test, a grep
that must return nothing), *Refs* — and complete one item at a time, ticking
it only when its Accept gate and the full build pass. State "done when"
before starting, including the honest negative arm ("works per-stream OR
documents why it stays single-stream"). If a needed fixture doesn't exist,
the honest terminal state is an explicit BLOCKED naming what a human must
supply — never a fabricated number. (This is PORTING.md §3 discipline; it
applies to feature work just as well.)

### 4.2 Run the gates that your change can affect

Fucina regresses in exactly two ways: it becomes **wrong** or it becomes
**slow** (`CONTRIBUTING.md`). Match the gate set to the blast radius — a doc
fix needs no benchmark; a tokenizer change needs the parity oracles but no
GEMM sweep; a kernel change needs both tracks, always.

| Gate | What it proves | Run when |
| --- | --- | --- |
| `zig build test` | every test root, native backend, no assets needed (§7.1) | always |
| `zig build test-fucina -Dbackend=scalar` | native agrees with the reference backend on the kernel/spec surface (the fucina root; model-golden forwards are native-only by design) | `src/backend/` or `src/exec/` changed — once, on final code |
| `zig build test -Dblas=none` | pure-Zig kernels unbroken | anything numeric near GEMM dispatch |
| `zig build arch-check` | layering intact (zero SCCs) | new files / imports |
| `zig build doc-check` | the doc index (`docs/README.md`) matches the docs-nav set; intra-doc links, anchors, and `src/<file>.zig:<line>` citations resolve; README's fetch pin matches the manifest | doc adds/moves/edits |
| `zig build snippet-check` | every runnable reference snippet still compiles and passes (§7.2) | any docs/reference edit; any public-API change |
| `zig build x86dot-check` | cross-ISA int8/quant dot parity + compile-only ISA legs | quant kernel / dot-arm changes |
| `zig build cuda-check` | CUDA provider still compiles (GPU-less machines) | exec/backend surface changes on GPU-adjacent code |
| `zig build metal-check` | Metal provider and its shim still compile (macOS hosts, no device needed) | exec/backend surface changes on GPU-adjacent code |
| `zig build bench-check` | every bench main still compiles (`addBench` registers each bench into the gate, so a new bench cannot land outside it) | bench/ or op-signature changes |
| Family parity oracles (`--tokenize`, logit parity, `--compare` batteries) | model behavior unchanged | anything touching a family |
| `tools/bench_gate.py` / `tools/opbench_gate.py` | speed did not regress (paired, median, CV-guarded) | any kernel/perf/hot-path change |

Anything under `src/backend/`, `src/exec/`, or a family's forward path needs
**both** tracks: correct-but-slower and fast-but-wrong are both regressions,
and neither is accepted alone. The single exception — a speed cost that is
the unavoidable price of a real correctness fix — must be stated explicitly
with before/after numbers.

### 4.3 Benchmark before "done"

A kernel/perf change is not done until measured. Bench in
`-Doptimize=ReleaseFast`, built natively (no `-Dtarget`) on the benchmarking
machine; validate in Debug/ReleaseSafe. The protocol, thermal discipline,
and the paired-gate tooling live in [BENCHMARK.md](BENCHMARK.md) — one
command in practice:

```sh
tools/fetch_refs.sh --build
python3 tools/bench_gate.py --models qwen3-0.6b-q6_k --tasks prefill,decode
```

Perf work respects the parity ratchet: an optimization that flips a single
token is reverted, not tolerance-adjusted. Scheduling-only changes are
proven bitwise; a change that alters floating-point summation order needs a
fresh tolerance argument against the pinned oracles. And remember the
profiler is the completeness oracle parity cannot be: parity passes even
when your fast arm is missing and dispatch fell to a slow path —
profile-confirm the hot path is armed.

### 4.4 Docs are part of the change

A public-API change updates the [reference](reference/00-index.md), one
chapter file per chapter under `docs/reference/`, and its snippets are
tests (`zig build snippet-check`; the authoring contract is §7.2). A new doc gets
a `<!-- docs-nav: ... -->` header (the sidebar placement
`tools/gen_docs_site.zig` reads) and a row in [docs/README.md](README.md),
the doc index; `zig build doc-check` verifies the index and the nav set
stay in sync. Write docs as timeless reference — what exists and how it
behaves; no dates (benchmark snapshots excepted), no development
narrative.

Tests follow the sibling convention: behavior in `<name>_tests.zig`, a
forwarding `test { _ = @import("<name>_tests.zig"); }` stanza in the
production file — and note the trap that a new `src/ag/` submodule must be
referenced from `ag.zig`'s test block or its sibling tests silently never
run. Always-passing tests must not print to stderr (route success-path
diagnostics through the root's `testlog` gate, e.g.
`apps/facedetect/testlog.zig`); asset-dependent suites skip cleanly
(`error.FileNotFound` → `error.SkipZigTest`, env-gated parity suites) so the
default `zig build test` is green with no assets.

### 4.5 Report what you did

In the PR or commit notes: the exact commands run, the machine and backend
configuration (CPU, OS, threads, `-Dblas`/`-Dbackend` flags), the model and
quantization used, and any failures or skipped suites. "Tests pass" without
the machine and the commands is not a report. If the change ports or adapts
third-party code, name the source, verify MIT compatibility, and credit it
in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) — uncredited ports are
treated as bugs.

## 5. Honest completion

Three habits keep the project's claims trustworthy, and they are part of the
method:

- **Claims name their pipeline.** Inference parity does not certify training
  parity; export interop is its own gate. Say which of the three a result
  covers.
- **Negatives are results.** A tried-and-reverted lever is recorded with its
  measured reason ("do not re-try without new evidence"); a documented
  negative can be a valid completion of a plan item. Symmetrically, re-verify
  any recorded premise against the current tree before building on it.
- **BLOCKED beats fabricated.** Missing fixture, missing hardware, missing
  reference — say so and name what a human must supply. The parity method
  only works because nobody invents its numbers.

## 6. API stability and deprecation

Fucina is pre-1.0 and versioned `0.MINOR.PATCH`: a MINOR release may change
public API, and every change that does is listed in `CHANGELOG.md` under that
release with the one-line rewrite for each affected call form. There is no
alias step before 1.0: a rename or move lands directly, the tree is migrated
in the same commit, and the CHANGELOG entry is the migration guide. The
public roots therefore never carry a `Deprecated:` declaration; `grep
'Deprecated:' src` returning nothing is part of the contract.

Module stability tiers are declared in each module's doc comment and
summarized in the CHANGELOG: **stable** modules (`tensor`, `ag`, `exec`,
`gguf`, `weights`, the serving contract) always get a CHANGELOG rewrite;
**experimental** modules (`es`, `ptqtp`, `speculative`, research features
under `models/`) may change with a CHANGELOG entry only.

CHANGELOG entries describe consumer-observable changes: public spellings,
build options, environment variables, on-disk formats, behavior. Internal
layout changes (file moves, module splits, private renames) stay in
`git log`; when a release contains many, the changelog carries at most one
short "Internal reorganization" paragraph naming the areas.

## 7. Test layout, doc snippets, and CI

### 7.1 Test organization

The default home for tests is a **sibling `*_tests.zig` file** next to the
production file it covers: `exec.zig` ↔ `exec_tests.zig`,
`src/models/text/tokenizer.zig` ↔ `src/models/text/tokenizer_tests.zig`,
and so on. Two sanctioned variations exist, each with a stated reason:

- **Directory suites** for a surface too large for one sibling:
  `src/ag/tensor_tests/` holds one file per op domain plus a shared
  `util.zig` helper, all forwarded from `src/ag/tensor.zig`'s test block.
  `arch-check` treats every file under a `<name>_tests/` directory as a
  test file, forwarded transitively.
- **Inline `test` blocks** where the tests exercise file-private symbols
  that a sibling cannot reach (`parallel.zig` documents this split in
  `parallel_tests.zig`'s header), or where the module is small enough that
  a sibling would be ceremony (`caching_allocator.zig`, the serving
  transport files).

A subdirectory facade forwards its children instead: `src/optim.zig`
forwards `src/optim/`, and its tests live in the root-level
`optim_tests.zig` covering the whole subtree. Whatever the shape, the
production file pulls its tests in with a forwarding stanza, so analyzing
the production file analyzes its tests:

```zig
test {
    _ = @import("exec_tests.zig");
}
```

Module roots forward everything: `src/fucina.zig` ends in a `test` block
referencing every submodule (`_ = dtype; _ = exec; …`), and `src/models.zig`
does the same for every family and helper, so one `addTest` per root
reaches the whole tree. `zig build arch-check` enforces the forwarding: a
sibling test file that no non-test file imports fails the gate, so a
forgotten stanza cannot silently drop a test file from `zig build test`.

`zig build test` runs every test root, each compiled as its own test binary
with the same option set as the corresponding executable: `src/fucina.zig`
(the core, with `build_options`), `src/models.zig` (the LLM/ASR stack,
imports `fucina`), `src/serving.zig` (the serving transport, imports
`fucina` and `fucina_models`), and the example roots wired in `build.zig`
(lmserve, nam, parakeet, omnivoice, locate_anything, facedetect,
voiceagent, nanochat), each also reachable alone through its solo step
(`test-fucina`, `test-models`, `test-serving`, `test-<example>`).

Every root passes with no model assets present. Suites that need external
material skip themselves cleanly rather than fail: the OmniVoice parity
suites gate on `OMNIVOICE_PARITY`
([§2.6](reference/02-toolchain-build-and-project-wiring.md#26-runtime-environment-variables)); asset-dependent tests
(facedetect goldens, the GGUF re-emit byte-identity test, tokenizer-parity
fixtures, NAM training goldens) translate `error.FileNotFound` into
`error.SkipZigTest`; GPU-dependent tests (`src/models/gemma/moe_tests.zig`)
skip unless the build has a GPU provider *and* a device is actually
present. Tests for **opt-in build features** follow the same discipline
through the feature's comptime flag: every
`src/models/text/llguidance_tests.zig` case is guarded on the flag — the
enabled-path cases open with
`if (!models.text.llguidance.enabled) return error.SkipZigTest;`, and one
disabled-build case inverts the guard to assert `error.LlguidanceNotEnabled`
— so the same test root compiles and passes under any flag combination and
gains coverage — never failures — when the flag is on.
Per `CONTRIBUTING.md`, numeric changes must additionally be green under
`-Dbackend=scalar` and `-Dblas=none` — the scalar backend is the reference,
and native must agree with it.

### 7.2 Doc snippets are tests too

`zig build snippet-check` extracts every runnable ```zig block from the
reference chapters under `docs/reference/` — any fenced block containing a
column-0 named `test "..."` declaration — into a generated test root and
runs it against the real `fucina`/`fucina_models` modules with the build's
option set (`tools/gen_snippet_tests.zig`). Authoring contract: snippets
assume an implicit prelude (`std`, `fucina`, `models = @import("fucina_models")`,
`optim = fucina.optim`; entries a snippet declares itself are not
re-emitted); a `<!-- snippet: helper -->` comment on the line before a
non-test fence marks a definition block (an Op/Spec/fn the prose
introduces) prepended to every later snippet in the same chapter file; a
`<!-- snippet: skip -->` comment excludes a test-shaped block that cannot
run hermetically. Illustrative fragments (signature blocks, bare `test {`
stanzas, asset-dependent `fn` examples) are ignored automatically. A
snippet for an opt-in build feature stays RUNNABLE, not skip-marked: it
opens with the feature's comptime-flag guard (e.g.
`if (!models.text.llguidance.enabled) return error.SkipZigTest;`), so
`snippet-check` compiles it under every flag combination and executes it
exactly when the enabling `-D` flag is passed.

### 7.3 Continuous integration (`.github/workflows/ci.yml`)

CI runs on pushes to `main` and on every pull request, on a two-OS matrix
(`fail-fast: false`): `ubuntu-latest` (x86-64) and `macos-15` (arm64 —
pinned rather than `-latest`, bumped deliberately). Zig 0.16.0 is installed
via `mlugg/setup-zig@v2`. Steps, in order:

1. `zig build test` — native backend (Accelerate on macOS, no BLAS on Linux,
   per the `-Dblas` default);
2. `zig build` — all executables compile;
3. `zig build bench-check` — the bench-check set compiles (bench mains are
   reachable only through their run steps, so nothing else in the
   build graph exercises them);
4. `zig build arch-check` — import-graph gate;
5. `zig build doc-check` — doc index, link, and citation gate;
6. `zig build snippet-check` — runnable-snippet gate (§7.2);
7. `zig build x86dot-check` — dot-kernel parity on the host ISA (x86 on
   ubuntu, NEON/sdot on macOS) plus the compile-only bit-rot legs;
8. `zig build test -Dbackend=scalar` — ubuntu only (the reference backend);
9. `zig build test -Dblas=none` — macOS only (pure-Zig native kernels,
   complementing the Accelerate run in step 1);
10. `zig build test -Dllguidance=true` + `snippet-check -Dllguidance=true` —
    ubuntu only (the runner image ships cargo): un-skips the flag-gated
    llguidance tests and snippets (§7.2), keeping the extern ABI, the cargo
    build, and the Rust-staticlib link from bit-rotting behind a green
    default build — and continuously proving the Linux link of that
    staticlib;
11. `zig build cuda-check` — ubuntu only: the compile-only CUDA-provider
    legs (fucina + models test roots and the PTX generator, cross-compiled
    for x86_64-linux-gnu, not run; no CUDA SDK involved);
12. `zig build metal-check` — macOS only: the compile-only Metal-provider
    legs (fucina + models test roots with the Objective-C shim and the
    Metal/Foundation frameworks linked, not run; no device needed).

Between the matrix and the conditional legs, every backend combination that
can run on stock CI hardware is covered: native+BLAS, native without BLAS,
and scalar, on both ISAs' unit-test surface, plus the opt-in llguidance
feature on Linux. Neither runner has a usable GPU, so both GPU providers are
covered at the compile level only: `cuda-check` on ubuntu and `metal-check`
on macOS keep the provider arms, their shims, and the model tier's provider
surface from bit-rotting behind a green `-Dgpu=none` build, while the
device-gated conformance suite (`src/backend/gpu_conformance.zig`) still
runs only on GPU hardware (`docs/GPU-OFFLOAD.md`). CPU dot ISA arms that CI
cannot execute (AVX-VNNI, AVX512-VNNI, smmla) are covered by the
compile-only legs and attestation records in `src/x86dot_check.zig`.
