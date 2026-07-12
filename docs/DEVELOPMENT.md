# DEVELOPMENT — how to build on Fucina

The working method for new development on this codebase — features, kernels,
model families, training machinery — written for the next contributor,
human- or agent-driven. `AGENTS.md` is the entry point (toolchain, commands,
repo map, house rules); [PORTING.md](PORTING.md) is the method for ports
specifically; [REFERENCE.md](REFERENCE.md) is the API contract. This file is
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

A band may depend only on bands at or below it: apps (`examples/`, `tools/`,
`bench/`) → llm (`src/llm/`) → facade (`src/fucina.zig`) → autograd/training
(`src/ag/`, optim/es/lora/persistence) → tagged ops (`src/tagged.zig`) →
exec runtime (`src/exec/`) → backends (`src/backend/`) → tensor/storage/dtype.
`fucina_llm` files import the `fucina` *module* (public surface plus
`fucina.internal`), never individual `src/*.zig` files.

*Enforced by:* `zig build arch-check` — the production import graph must have
zero strongly-connected components (AST-based, test-aware). Band *direction*
is review-checked against the layer table in
[ARCHITECTURE.md](ARCHITECTURE.md): production layer inversions are bugs,
full stop.

*Violation:* exec importing ag; an example importing `src/tensor.zig`
directly; family-specific logic inside `src/exec/` (see §1.8).

### 1.2 Eager and local — no graph compiler (review-only)

No global graph object, no fusion pass, no lazy evaluation, no planner. Every
op validates, allocates through `ExecContext`, and runs a kernel immediately.
This is a deliberate design stance, not debt. An app-level compiled replay is
fine *inside an example* (`examples/facedetect/graph.zig` is the precedent);
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
`ExecContext`/`Runtime` supplies. The one sanctioned exception is the
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

Reusable engines other `src/llm` consumers should import go in
`src/llm/<family>/`; single-purpose parity ports and their DSP/IO plumbing
stay example-local in `examples/<name>/`; generic helpers stay flat in
`src/llm/`; family-specific kernel orchestration lives in the family over
the `fucina.internal` seam, never inside the generic exec runtime; shared
cross-family scheduling lives once in exec and is re-exported
(`exec/moe_chain.zig` is the pattern). New core tensor ops belong in the
exec/backend bands — but a good port usually needs none: nanochat is
entirely example-local over the public facade.

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
Anything numeric runs the scalar leg before merge (once, on the final code —
it is slow by design). Everything integer is bit-exact across architectures;
float tile kernels document association-order tolerance instead.

## 2. Check before you build

The most common failure mode of a capable contributor on this codebase is
rebuilding something that exists. Search this table first; the § pointers go
into [REFERENCE.md](REFERENCE.md), whose snippets are machine-verified.

| You need… | It exists as… | Where |
| --- | --- | --- |
| A contraction (matmul, batched, multi-index) | `einsum` — THE contraction engine; `dot` is its special case. Don't hand-roll permute+matmul chains. | §4.8–4.9 |
| A new pointwise op with autograd | `elementalUnary`/`elementalBinary` — supply scalar fwd/bwd fns, get a SIMD-chunked parallel op with a VJP. | §4.4 |
| Indexing / slicing / functional updates | `select`, `slice`, `sliceStep`, `indexSelect`, `gather`, `setSlice`, `setRows`, `indexAdd`, `scatterAdd`, `maskedScatter`, `where`, `oneHot` | §3.7, §4.17 |
| Reductions, scans | SIMD-promoted sum/mean/max/min/prod/logsumexp; `cumsum`/`cumprod` (+ `-Dvector-scan`) | §4.7 |
| Attention | `groupedAttention`: causal/bidirectional, sliding window, additive bias, sinks, ALiBi-style `max_bias`, f32/f16/q8 KV, saved-stats backward | §4.13 |
| RoPE | interleaved/half modes, partial rotary, freq-factor (YaRN-style) tables, inverse tables, hand-fillable `RopeTable` | §4.12 |
| Norms, softmax, losses | rmsNorm family (incl. fused mul/add/rope), layerNorm, groupNorm; softmax with scale/mask/sinks; `crossEntropyExt` (smoothing, ignore-index, accum scale), mse/huber/bce/kl/nll | §4.10–4.11, §4.15 |
| Vision / conv | channel-last conv2d (im2col GEMM + Winograd), conv1d/causal/transpose, pool2d, prelu, channelAffine, upsample, zeroPad2d — all with autograd | §4.14 |
| MoE | `routerTopK`, `moeExpertFfn`/`Batch`, `MoeRhs` packed containers, `moe_chain` scheduling, disk-streamed experts (`ExpertStore`) for models larger than RAM | §4.16, §4.18, §13.2 |
| Autograd machinery | seeded backward (`backwardWithGrad`), `noGrad`, activation checkpointing, `customVjp`, `gradcheck`; VJP inventory in §5.8 | §5 |
| Training | SGD/AdamW/Adam/Muon/APOLLO (torch-golden-parity), `OptimizerSet` param groups, LR schedules, clipping, `ParamRegistry`, LoRA adapters, ES (incl. ternary-native), 16-bit leaves with f32 grads + optimizer masters | §11 |
| Quantized weights | hot packed kernels (Q4_K/Q5_K/Q6_K/Q8_0/TQ2_0 + 2-bit expert tier), cold decode (IQ*, FP4…), byte-exact ggml encoders, PTQTP trit-planes | §10 |
| Persistence | GGUF read/write/transcode (byte-verbatim re-emit), safetensors, named state dicts with alias remapping, training-checkpoint directories, `export-gguf` (incl. LoRA merge) | §12 |
| LLM plumbing | `LinearWeight` (dispatch to BLAS/Metal/CUDA/quant kernels is *inside* — never hand-roll a linear), `gguf_meta` readers, KV cache (f16/q8_0) + crash-safe persistence, BPE + SPM tokenizers, sampler + `LogitProcessor` + llguidance constrained decoding, generic `Conversation` chat engine (+ `sendBatch`), speculative decoding cascade + grammar-constrained drafting, native MTP | §13 |
| Parallelism / infra | worker team + `parallelChunks` (this *is* the parallel-loop contract), `BufferPool`, `RhsLifetime` RHS caching, deterministic RNG, GPU offload gates | §6, §9 |

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
| LLM family, llama-shaped | `src/llm/qwen3/` | cleanest dense+MoE model, spec decode, trainer |
| LLM family, SPM tokenizer / sliding window / MoE engines | `src/llm/gemma/` | per-layer KV geometry, MoE engine with GPU arm |
| Hybrid/recurrent blocks | `src/llm/qwen35/` | Gated-DeltaNet over the same loader conventions |
| MLA / MTP / streamed-expert giants | `src/llm/deepseek4/` | compressed KV, hyper-connections, out-of-core experts |
| Non-autoregressive decoder | `src/llm/diffusion_gemma/` | two forward modes over one weight set |
| Pure-CNN vision port | `examples/facedetect/` | load-once models, BN-fold at load, byte-identical JSON goldens |
| VLM port | `examples/locate_anything/` | ViT tower + LM, custom RopeTable, MTP box decode |
| ASR / encoder stack | `src/llm/parakeet/` | the reusable-family precedent |
| TTS / codec port | `examples/omnivoice/` | codec parity, RNG parity, chunked streaming |
| Streaming DSP / effects | `examples/nam/` | streaming engines, format interchange, live IO |
| Training pipeline | `examples/nanochat/`, `examples/spirals.zig`, `examples/finetune.zig` | full pretrain→SFT→chat; minimal optimizer demo; LoRA on a real GGUF |
| HTTP/API frontend | `examples/lmserve.zig` | OpenAI-compatible mapping tables, SSE, backend matrix |

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
| `zig build test` | nine test roots, native backend, no assets needed | always |
| `zig build test -Dbackend=scalar` | native agrees with the reference backend | anything numeric — once, on final code |
| `zig build test -Dblas=none` | pure-Zig kernels unbroken | anything numeric near GEMM dispatch |
| `zig build arch-check` | layering intact (zero SCCs) | new files / imports |
| `zig build doc-check` | AGENTS.md doc index resolves | doc adds/moves |
| `zig build snippet-check` | every runnable REFERENCE.md snippet still compiles and passes | any REFERENCE.md edit; any public-API change |
| `zig build x86dot-check` | cross-ISA int8/quant dot parity + compile-only ISA legs | quant kernel / dot-arm changes |
| `zig build cuda-check` | CUDA provider still compiles (GPU-less machines) | exec/backend surface changes on GPU-adjacent code |
| `zig build bench-check` | every bench main still compiles | bench/ or op-signature changes |
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

A public-API change updates [REFERENCE.md](REFERENCE.md) — and its snippets
are tests (`zig build snippet-check` extracts every fenced block with a
column-0 `test "..."` and runs it against the real modules). The authoring
contract is REFERENCE.md §2.7: implicit prelude (`std`/`fucina`/`llm`/
`optim`), `<!-- snippet: helper -->` for prose-introduced definitions,
`<!-- snippet: skip -->` only for genuinely non-hermetic blocks; opt-in
feature snippets stay runnable behind their comptime-flag guard. New docs
get a row in `AGENTS.md`'s doc index (doc-check verifies it). Write docs as
timeless reference — what exists and how it behaves; no dates (benchmark
snapshots excepted), no development narrative.

Tests follow the sibling convention: behavior in `<name>_tests.zig`, a
forwarding `test { _ = @import("<name>_tests.zig"); }` stanza in the
production file — and note the trap that a new `src/ag/` submodule must be
referenced from `ag.zig`'s test block or its sibling tests silently never
run. Always-passing tests must not print to stderr (route success-path
diagnostics through the root's `testlog` gate, e.g.
`examples/facedetect/testlog.zig`); asset-dependent suites skip cleanly
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
