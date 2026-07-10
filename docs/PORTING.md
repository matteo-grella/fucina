# PORTING — how a port earns its place

Every model family and engine in this repo (Qwen3, Gemma 4, Qwen3.5,
DiffusionGemma, Parakeet, OmniVoice, NAM) was ported from a reference
implementation with the same method. This file is that method, written for
the next port — human- or agent-driven. The one-line version: **you don't
optimize what you can't verify**, so the verification oracle is built
first, parity is closed stage by stage behind mechanical gates, and only
then does performance work begin, with the parity gates frozen as a
non-regression ratchet.

## 1. Pin the reference kit before writing any code

- **Pin the reference at an exact commit** and build it locally under
  `refs/` (see `tools/fetch_refs.sh` for the existing pins and the
  convention). If the reference is a moving target — a draft PR, an active
  branch — save the diff, record the exact fetch/build commands for the
  oracle binary, and write down how to regenerate any scratch parity
  inputs, because they die.
- **Choose the smallest variant that exercises every code path you must
  port** (the Parakeet port anchored on a 110M hybrid precisely because it
  has *both* decode heads), fix one input, and record the reference's
  output for it as the hard target.
- **Write the contract from the actual artifact, not the paper.** Parse
  the real weights file and verify every geometry assumption against its
  metadata; papers, model cards, and your own task notes will be wrong
  about layer counts, kernel sizes, padding modes, and tensor names. Make
  the config fully metadata-driven so sibling sizes load for free.
- **Deep-read the reference into cited maps** — one per subsystem (model
  math, codec/DSP, orchestration), every claim anchored `file:line` into
  the pinned clone. The maps are the port spec; treat any anchor as a
  dated snapshot and re-check it before acting on it.
- **Enumerate the numeric traps as a ranked checklist**: tie-break
  directions (first-wins vs last-wins), accumulation order and width,
  epsilon placement, `>` vs `>=` thresholds, RNG draw counts per step, and
  every approximated primitive the reference ships (lookup-table GELU,
  a library's non-libm tanh, a CUDA-aligned RNG, a resampler's exact
  window). Parity means reproducing these idiosyncrasies, not the
  textbook function.

## 2. Build the oracle before the port

- **Mirror the reference's own dump surface** (stage names, dump format,
  debug flags) instead of inventing a comparison surface. If it has none,
  instrument it minimally: env-var-gated hooks (zero overhead unset),
  kept as a re-applyable patch under `tools/ref-patches/` so dumps are
  regenerable from a fresh clone; force the reference onto its
  deterministic CPU path when dumping; and verify the instrumented binary
  still produces byte-identical final output — dumps must not perturb the
  thing they measure.
- **Dumps are self-describing** (magic + dtype + dims header, header
  validated against payload size) and each records its memory layout
  explicitly — layout confusion is a classic silent-parity killer.
- **Deterministic mode is the first oracle** (greedy, zero RNG draws).
  Only after it passes, port the RNG exactly — algorithm, counter layout,
  draws-per-step accounting — so seeded stochastic runs become
  token-exact too. The draw count is itself a gate.
- **Hybrid architectures need intermediate oracles planned up front**:
  when new sublayers interleave with known-good ones, final-logit parity
  is unreachable until everything works, so per-sublayer reference dumps
  must exist before the port starts.
- **Verify your verifiers.** A harness that prints PASS/FAIL is not a
  gate until it exits nonzero on failure; accept criteria must be
  behavioral (a name-grep passes on the wrong code); structural checkers
  pass on code that doesn't compile.

## 3. Stage the port behind mechanical gates

- **Pipeline-ordered stages, one parity gate each**, exposed as exit-code
  `--compare <stage> <dump>` modes in the port's own CLI (see
  `examples/parakeet.zig` and `examples/omnivoice.zig`). Intermediates
  are numeric checkpoints whose only job is to localize the first
  divergent stage; the final gate is exact match of the discrete output.
  A stage is done when its gate passes, tests are green on both backends
  (`-Dbackend=scalar` is the reference), and the work is committed.
- **Order stages so the reference feeds later oracles**: port the codec
  *decoder* before the *encoder* (the reference encoder produces the
  oracle inputs your decoder consumes); close an end-to-end exact-output
  loop as early as possible via the simplest head, so every later change
  has a full-pipeline oracle.
- **Executable plans have three parts per item** — *Do* (imperative,
  with code anchors), *Accept* (a mechanical gate: an exit code, a grep
  that must return nothing, a named test), *Refs* — and an agent loop
  executes exactly one unchecked item per iteration, ticking it only when
  the Accept gate and the full build pass.
- **State "done when" before starting, with an adjudication arm**: a
  documented negative ("works per-stream OR documents why it stays
  single-stream") is a valid completion. And **never fabricate parity**:
  if a fixture is missing, the honest terminal state is an explicit
  BLOCKED naming what a human must supply — not an invented number.

## 4. The tolerance policy

Tier the gate by what sits upstream of the checkpoint:

| Checkpoint | Gate |
| --- | --- |
| Discrete outputs (token ids, transcripts, codec codes, argmax) | **exact equality** |
| Stages computed purely in f32 | tight max-abs (~1e-4) |
| Anything downstream of f16/quantized weights | cosine (>= 0.9999x) |

Never soften the discrete gate to a similarity score, and never demand
bitwise floats mid-pipeline — the method rests on numeric drift not
flipping the argmax. Corollaries, each learned the hard way:

- **Where a discrete decision rides on ulps**, bit-exactness requires
  reproducing the shipped binary's actual arithmetic — its f16
  intermediates, fp contraction, vendored BLAS, system libm — and such a
  contract is scoped to the platform whose arithmetic you matched.
- **Know what parity cannot promise.** A dtype with per-GEMM activation
  quantization can never be cross-implementation token-exact (last-ulp
  drift flips int8 rounding and compounds); prove quantized quality by
  showing your outputs sit at the same distance from the f32 truth as the
  reference's own. Even f32 cross-implementation equality is
  corpus-dependent — pin golden inputs rather than claiming universality.
  The only universal guarantee is same-build determinism.
- **For numerically chaotic models**, run the reference against itself
  and require your implementation to sit inside the reference's own
  disagreement envelope — do not chase a bar the oracle itself cannot
  meet. Where upstream publishes a cross-implementation tolerance, gate
  tighter than it and report the achieved margin.
- **Parity-critical control paths are verbatim scalar host code** —
  schedules, CFG/log-softmax, top-k, sampling bookkeeping. They are never
  the bottleneck, and optimized kernels' reassociation breaks near-tie
  decisions.
- **The tolerance is part of the port's contract.** A refactor or
  optimization that moves a parity result is reverted; the tolerance is
  never loosened to make a gate pass.

## 5. The LLM parity ladder

For language models specifically, climb in fixed order — each rung
isolates the next from contamination:

1. **Tokenizer, token-ID-exact** against the reference tokenizer, via a
   weights-free oracle mode (`qwen3 --tokenize`) over adversarial
   fixtures: code, unicode classes, emoji ZWJ sequences, whitespace runs,
   empty input.
2. **Logits from explicit raw token ids**, so tokenizer differences
   cannot masquerade as model bugs (`--logits-out` +
   `tools/llama_logits.cpp`).
3. Only then trust **generation**, acceptance rates, and benchmarks.

Loader discipline: resolve every expected tensor by name with shape and
dtype-class checks, **and assert the resolved count equals the file's
total tensor count** — total coverage surfaces variant drift immediately,
and the honest response to a non-conforming variant is a recorded
deferral, not a loosened check. When inventing a GGUF scheme for a
non-GGUF upstream: keep tensor names as verbatim upstream state-dict
keys, namespace config under `<arch>.*` and read every hyperparameter
from it, honor per-tensor dtype, and bake preprocessing constants (mel
filterbanks, windows) into the file so the port is load-and-matmul, not a
DSP reimplementation.

## 6. Interop is proven in both directions

Import parity is half a port. The other half: **the upstream
implementation must accept your exports** — upstream's engine loading a
Fucina-trained `.nam` and cross-rendering it, llama.cpp serving a
Fucina-merged GGUF — because that is the gate every ecosystem consumer
actually runs. Practices that make it hold:

- **Tolerant reader, canonical writer** — accept every legacy spelling
  upstream tolerates, emit only the modern shape, and validate the writer
  against the *strictest* consumer in the ecosystem, not the mainstream
  one.
- **Byte-verbatim retention** is the strongest round-trip guarantee:
  when a container cannot express the source format's structure, embed
  the original file verbatim next to the derived tensors and `cmp` the
  round trip (the GGUF writer's metadata passthrough is the same
  pattern).
- **Weight order is the classic port-bug surface**: derive the flat
  consumption order line-by-line from the *consuming reader's* code,
  cross-check against the writer, validate with a total-count formula
  reproducing upstream's own files — then re-encode that formula as a
  load-time check, because upstream's end-of-stream asserts vanish in
  release builds.
- **Keep a numbered deviations register.** Every intentional divergence
  from upstream gets an entry with its justification and measured
  magnitude; code and tests reference deviations by number; later parity
  audits classify findings against the register. When upstream's own
  writer and reader disagree, match the reader the ecosystem runs, and
  record the discrepancy as an upstream bug so it isn't mistaken for
  yours.

## 7. Performance after parity — and parity as the ratchet

Perf work starts only when parity is closed, and the exact gates become a
hard non-regression ratchet: every optimization re-passes the full parity
battery on every supported format and both backends; an optimization that
flips a single token is reverted, not tolerance-adjusted. With that:

- **Gate on the full configuration matrix** when an effect can vary by
  dtype, BLAS provider, or model pairing — a single-row spot check has
  shipped a hard regression here before.
- **The profiler is the completeness oracle parity cannot be**: parity
  tests pass even when a fast kernel arm is missing and dispatch falls to
  a scalar path. After wiring a new format or target, enumerate which
  kernel variants the real dispatch takes and profile-confirm the hot
  path is armed.
- **Keep a documented-negatives register** — every tried-and-reverted
  lever with its measured reason and "do not re-try without new
  evidence" — and, symmetrically, re-verify any recorded premise on the
  current tree before building on it: stale premises in old briefs have
  nearly shipped regressions twice.
- **A parity fix that touches a hot kernel re-runs the perf gate.**
  Correctness and speed are paired tracks (`CONTRIBUTING.md`), especially
  for audit-driven fixes — one audit's "safer tanh" hid a multi-x
  regression for weeks.
- **Scheduling-only changes are proven bitwise**; any change that alters
  floating-point summation order needs a fresh tolerance argument against
  the pinned oracles — and if that argument is expensive, defer the
  change until profiling shows it matters.

## 8. A new ISA is a port too

Porting Fucina itself to a new architecture follows the same method, with
its own traps:

- **Phase 0 is deliberately correct-but-slow**: comptime-guarded portable
  bodies for every arch-specific primitive so the whole engine compiles,
  then run the existing parity suite on the new target before writing any
  fast kernel — the oracle is ISA-agnostic and travels.
- **Write a three-tier portability map first**: portable as-is /
  principle transfers but must be re-expressed / architecture-specific
  with named instruction analogs. A compiling fallback in tier two does
  not deliver the win, and every tuned constant is a host fact to
  re-derive — one knob at a time, only after parity is green.
- **Never trust an emulator until probed per ISA tier with semantic
  asserts**: emulators can execute unsupported instructions silently
  wrong rather than faulting. Pin a validated emulation recipe, and
  record which tiers no available emulator can execute.
- **For arms no local machine can run**: compile-only build legs stop
  bit-rot, and a dated per-arm execution-attestation table records where
  each arm has actually *executed* (see `src/x86dot_check.zig`) —
  compilation proves nothing about numerics. Know which arm each test
  run really exercised: build mode and feature gates silently swap arms,
  and a green sibling arm proves nothing across different saturation
  semantics.
- **Retired scalar kernels live on as public bit-exactness references**,
  and the policy states explicitly where cross-arch bitwise identity is
  required (everything integer) versus documented association-order
  tolerance (float tile kernels).

## 9. Honest claims, audited

Inference parity does not certify training parity — the training pipeline
is a separate parity target with its own audit, and the audit should
expect the docs to overclaim: correcting them is part of closing it.
Every claim names the pipeline it covers (inference / training / export),
plans become design records with dated outcome addenda rather than
rewrites, and a spike that lives on an experiment branch is recorded with
its commit and a "re-baseline first" note so no future session needs
archaeology. That discipline is why this file could be written at all.
