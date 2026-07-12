# Chapter 16 — The craft: building a library that can be trusted

*Part VI — The craft*

Fifteen chapters ago this course started with an empty file and a dtype enum. By [Chapter 15](15-training-llms-on-cpu.md) the same codebase was fine-tuning quantized language models on a laptop. Somewhere between those two points, a question changed: at first it was *"how do I build this?"* — now it is *"how do I keep this true?"* A tensor library is a hundred-odd source files where a single wrong sign produces a model that trains slightly worse, a single reordered addition flips a token three layers downstream, and a missed SIMD arm silently costs most of your throughput while every test stays green — Fucina's own x86 record shows a 7.2x self-speedup on one shape from arming kernels that had been falling back to scalar (`docs/BENCHMARK.md`, q8_0 pp256, i9-13950HX). None of these failures announces itself. All of them compound.

This chapter is about the machinery that catches them. Not "clean code" as an aesthetic — clean code as *physics*: rules a machine checks on every push, oracles that make "correct" a mechanical question, benchmark protocols that make "fast" a falsifiable claim, and a written culture of recording losses, negatives, and gaps as plainly as wins. Everything here is transferable. Fucina is the worked example, but the method — build the oracle first, make the boring version the specification, ratchet the gates, date the numbers, write down what you rejected and why — applies to any project where being wrong is quiet and being slow looks identical to being right.

One framing sentence organizes everything that follows. It opens the "two regression tracks" section of `CONTRIBUTING.md`:

> Fucina regresses in exactly two ways: it becomes **wrong**, or it becomes **slow**.

Every gate, oracle, and protocol in this chapter exists to catch one of those two, and every change is tested "against the failure modes it can realistically affect — no more, no less" (`CONTRIBUTING.md`). A doc fix needs no benchmark; a tokenizer change needs the parity oracles but not a GEMM sweep; a kernel change needs both tracks, always. Hold that sentence; the rest of the chapter is its expansion.

## 16.1 An architecture you can draw — and check

Chapter after chapter, this course built Fucina bottom-up: dtypes and storage ([Chapter 3](03-tensors-from-scratch.md)), tags ([Chapter 4](04-axes-with-names.md)), the exec runtime ([Chapter 5](05-the-operation-library.md)), backends ([Chapter 6](06-going-fast-on-cpus.md)), autograd ([Chapter 7](07-autograd.md)), training ([Chapter 8](08-training.md)), the LLM stack (Chapters [11](11-model-files-and-quantization.md)–[15](15-training-llms-on-cpu.md)). That order was not a narrative convenience. It is the library's actual dependency structure, written down as a table in `docs/ARCHITECTURE.md`:

```text
Top-down; a band may depend only on bands at or below it:

| Band | Contents |
| --- | --- |
| apps        | examples/**, tools/**, bench/**, src/bench_raw.zig, src/x86dot_check.zig |
| llm         | src/llm.zig, src/llm/** (the fucina_llm module) |
| facade      | src/fucina.zig (the fucina module root) |
| ag + training/serialization | src/ag.zig, src/ag/**, src/optim.zig, src/es.zig, src/gguf.zig, src/lora.zig, … |
| tagged      | src/tagged.zig (tag-ops library) |
| exec        | src/exec.zig, src/exec/** (eager runtime) |
| backend     | src/backend.zig, src/backend/** (numeric kernels) |
| tags        | src/tags.zig (comptime tag algebra) |
| tensor      | src/tensor.zig (raw tensor) |
| primitives  | src/thread.zig, src/parallel.zig |
| core        | src/dtype.zig, src/storage.zig, src/accelerator.zig, src/rng.zig |
```

One rule: **a band may depend only on bands at or below it.** Read the consequences off the table. The tensor does not know autograd exists. The backend does not know models exist. The exec runtime never contains family-specific logic — when Gemma's MoE and Qwen's MoE both needed batched expert scheduling, the shared scheduler landed *once* in `src/exec/moe_chain.zig` and both families import it downward, rather than exec growing an `if (model == .gemma)` upward. And the `fucina_llm` module imports the `fucina` *module* — the public surface plus the explicit `fucina.internal` escape hatch — never individual `src/*.zig` files (`docs/ARCHITECTURE.md`, *Layering And Enforcement*).

This is what "clean code" means in a performance library: not short functions or fashionable naming, but a dependency direction you can state in one sentence and verify mechanically. Lower layers stay reusable because they are ignorant; upper layers stay replaceable because nothing below reaches up into them.

The layering has teeth even at the API boundary. Fucina's raw f32 tensor — the workhorse of Chapters 3–6 — is deliberately *not* part of the public surface, and the rule is enforced by the compiler itself, in `src/fucina.zig:28-42`:

```zig
// Deliberately NO public `RawTensor` root export. Raw f32 tensors are an INTERNAL
// runtime/backend detail, not a stable public API — the no-grad `Tensor` facade
// has negligible forward overhead, so model/example code carries
// `fucina.Tensor(spec)` end-to-end. In-tree raw naming uses `fucina.internal.RawTensor`;
// microbenchmarks use `bench_raw.RawTensor`.
comptime {
    // Anti-regression guard: re-exporting the raw tensor type at the
    // PUBLIC ROOT is a COMPILE ERROR. This fires on any build that analyzes the
    // module root (every test/example/tool), not just `zig build test`. `internal`
    // and `bench_raw` are unaffected (this only inspects the root's own decls).
    if (@hasDecl(@This(), "RawTensor")) @compileError(
        "fucina.RawTensor must not be exported at the public root; raw tensors are internal. " ++
            "Use fucina.internal.RawTensor (in-tree raw naming) or bench_raw.RawTensor (microbench).",
    );
}
```

> **Zig note** — This is a `comptime` block at file scope: it runs during compilation of the module root, on every build that analyzes it. `@hasDecl(@This(), "RawTensor")` asks "does this file declare a public `RawTensor`?" and `@compileError` fails the build with a written explanation if anyone ever adds one. A design rule ("the raw layer stays sealed") has been turned into something no future contributor can violate absent-mindedly — the reviewer is the compiler, and the review comment is the error message. Note the escape hatch is *named in the error*: good guards tell you the sanctioned alternative, not just "no".

> **Zig note** — The top two bands are also *module* boundaries, not just directories. `build.zig` wires two library modules — `fucina` rooted at `src/fucina.zig` and `fucina_llm` rooted at `src/llm.zig` — and the llm module's only library import is the `fucina` module root (`docs/ARCHITECTURE.md`, *Build And Verification*). Two Zig rules give the boundary teeth: named module imports like `@import("fucina")` resolve only if the build declaration hands them over, and every file must belong to exactly one module. So "llm files never import individual `src/*.zig` files" is not a convention — an `@import("../exec.zig")` from inside `src/llm/` is a compile error ("file exists in modules 'fucina' and 'fucina_llm'"), because `src/exec.zig` already belongs to the `fucina` module. The build graph is itself one of the enforcement layers.

## 16.2 What the machine enforces, and what it can't

A layer diagram that lives only in a document rots. Fucina's answer is `zig build arch-check`, which runs `tools/check_import_graph.zig` over the production import graph on every CI push. Its module header states the contract exactly (`tools/check_import_graph.zig:1-15`):

```zig
//! Production import-graph checker for Fucina.
//!
//! ... This tool
//! enforces the stricter production invariant: non-test `src/**/*.zig` local
//! imports must have no nontrivial strongly-connected components.
//!
//! Test awareness inside production files: an `@import` is counted only when
//! it is reachable from production code. Skipped are (a) imports inside `test`
//! declarations, and (b) imports inside non-pub file-scope decls that no
//! production decl references (e.g. a private test-only helper fn).
```

The tool parses every production source file, builds the local-import graph, runs Tarjan's strongly-connected-components algorithm over it, and fails if any nontrivial SCC exists — that is, if any group of production files imports each other in a cycle. Current output, as recorded in `docs/ARCHITECTURE.md`:

```text
production import graph: 105 files, 408 edges, 0 SCCs
```

Two subtleties make this checker honest rather than merely strict, and both are worth stealing for your own projects:

1. **It is test-aware.** Fucina's test convention (§16.5) has every production file pull in a sibling test file via `test { _ = @import("exec_tests.zig"); }` — and the sibling imports the production file back. That is a 2-cycle, and a naive cycle checker would flag all 143 of them. The tool instead excludes `@import`s inside `test` declarations and inside private test-only helpers, so the *production* graph it checks is the graph that actually ships. A lint that cries wolf gets deleted; a lint that understands the codebase's idioms survives.
2. **It fails conservatively.** Files that fail to parse count *every* import (`tools/check_import_graph.zig:14-15`). When the tool is unsure, it errs toward flagging — the safe direction for a gate.

Now the precision that matters, because it is easy to over-claim: **arch-check enforces acyclicity, not band direction.** Zero SCCs proves no production file participates in an import cycle; it does not prove that `exec` never imports `ag` (a downward-pointing but band-violating edge would still be acyclic). The band-direction rule from the table in §16.1 is enforced in review — `docs/DEVELOPMENT.md` §1.1 states it plainly: band direction "is review-checked against the layer table in ARCHITECTURE.md: production layer inversions are bugs, full stop." (`docs/ARCHITECTURE.md` notes that a dependency-structure lint additionally checks the bands during development, but its configuration is not part of this tree — so from the repository's point of view, direction is a review invariant, not a machine gate.)

`docs/DEVELOPMENT.md` §1 is candid about this split in general. It lists the project's ten invariants in a fixed three-part format — the rule, *where it is enforced*, and *what a violation looks like* — and the ones with no mechanical gate are explicitly marked **review-only**: "nothing fails automatically when you break them, which is exactly why they are listed here." The enforcement lines are refreshingly concrete. The ownership invariant (§1.4) is enforced by "the testing allocator (leaks fail tests), the BufferPool's `outstanding == 0` teardown assert, and the `undefined` tripwire in Debug" — three independent mechanical detectors for one rule. The kernel contract from [Chapter 6](06-going-fast-on-cpus.md) — kernels never allocate (§1.5), validate-then-call-unchecked (§1.6) — is review-only, and §1.6 explains why review must be strict there: "ReleaseFast drops safety checks, so a kernel that only behaves because Debug catches it is broken — prove invariants, don't use checks as logic."

That is the transferable lesson, in two halves: enforce what you can, and for what you can't, write the rule down *with its violation signature* and label it as unenforced, so a reader knows which invariants are load-bearing documentation rather than checked fact — and knows what to grep for in review.

> **ML note** — Why does an ML library care this much about import direction? Because the alternative is the framework pathology where the tensor knows about the graph, the graph knows about the device runtime, the device runtime knows about serialization, and touching any one of them means understanding all of them. Fucina's backward pass ([Chapter 7](07-autograd.md)) can walk live tensors precisely because the tensor layer was built with no knowledge of autograd — the graph is layered *on top of* values, not woven *through* them.

## 16.3 The verification religion

Everything in this section descends from one sentence in `docs/PORTING.md`:

> **you don't optimize what you can't verify**, so the verification oracle is built first, parity is closed stage by stage behind mechanical gates, and only then does performance work begin, with the parity gates frozen as a non-regression ratchet.

An *oracle* is any independent source of truth you can compare against mechanically: a reference implementation, a slower-but-obvious algorithm, a mathematical identity, a pinned golden file. Fucina's discipline is that every claim of correctness is anchored to one, and the README states the inventory in a single breath (`README.md`, *What runs today*): "token-ID-exact tokenizers vs `llama-tokenize`, logit-parity oracles vs llama.cpp, byte-exact quantization encoders vs ggml, byte-identical GGUF re-emit." Let's collect the full arsenal, because as a *set* it is a reusable methodology.

### The scalar backend: a specification you can execute

The deepest oracle is inside the library itself. Fucina has exactly two CPU backends ([Chapter 6](06-going-fast-on-cpus.md)): **scalar** — plain loops, no SIMD, no cleverness — and **native**, the fast one. The scalar backend is not a fallback; it is the *executable specification*. `docs/DEVELOPMENT.md` §1.10:

> The scalar backend (`-Dbackend=scalar`) is the executable reference: native and scalar must agree, and `src/backend/parity_test.zig` holds them together.

A specification written as slow, obvious code beats a specification written in prose, because you can diff against it automatically. The policy even has numeric tiers: "Everything integer is bit-exact across architectures; float tile kernels document association-order tolerance instead" (same section). Here is the whole pattern in miniature — course code, not from the repo:

```zig
// Course code — the parity-twin pattern, self-contained.
const std = @import("std");

// A deliberately boring reference: the specification.
fn sumScalar(xs: []const f32) f32 {
    var acc: f32 = 0;
    for (xs) |x| acc += x;
    return acc;
}

// The "fast" version: 4 lanes at a time, tail handled scalar.
fn sumVector(xs: []const f32) f32 {
    const L = 4;
    var acc: @Vector(L, f32) = @splat(0);
    var i: usize = 0;
    while (i + L <= xs.len) : (i += L) {
        const v: @Vector(L, f32) = xs[i..][0..L].*;
        acc += v;
    }
    var total = @reduce(.Add, acc);
    while (i < xs.len) : (i += 1) total += xs[i];
    return total;
}

test "parity twin: the fast kernel never escapes the slow kernel's judgment" {
    var prng = std.Random.DefaultPrng.init(42);
    const random = prng.random();
    var xs: [103]f32 = undefined; // deliberately not a multiple of the lane width
    for (&xs) |*x| x.* = random.float(f32) * 2 - 1;
    const reference = sumScalar(&xs);
    const fast = sumVector(&xs);
    // A float kernel documents association-order tolerance; an integer
    // kernel would gate bit-exactly.
    try std.testing.expectApproxEqAbs(reference, fast, 1e-4);
}
```

Note the length 103: parity tests probe the *tail paths*, because kernels are correct on nice multiples of the lane width by construction and wrong at the edges by habit. And note the tolerance comment — it mirrors Fucina's real policy, where the choice between "bit-exact" and "documented tolerance" is itself part of the kernel's contract.

### The tiered tolerance policy

When the oracle is an *external* implementation, "must agree" needs a definition, and "within 1e-6 everywhere" is the wrong one. `docs/PORTING.md` §4 tiers the gate by what sits upstream of the checkpoint:

| Checkpoint | Gate |
| --- | --- |
| Discrete outputs (token ids, transcripts, codec codes, argmax) | **exact equality** |
| Stages computed purely in f32 | tight max-abs (~1e-4) |
| Anything downstream of f16/quantized weights | cosine (>= 0.9999x) |

And the two rules that keep the table honest: "Never soften the discrete gate to a similarity score, and never demand bitwise floats mid-pipeline — the method rests on numeric drift not flipping the argmax."

> **ML note** — This table is a compressed lesson in what neural-network numerics can and cannot promise. Floating-point addition is not associative, so two correct implementations that sum in different orders produce different low-order bits — demanding bitwise float equality across implementations is demanding they share bugs-for-bugs arithmetic. But a language model's *output* is an argmax over logits: a discrete choice. Small drift that doesn't flip the argmax is invisible; drift that does flip it is a real behavioral difference. So the discrete outputs gate exactly, the floats in between gate within justified tolerance, and quantized paths — where per-GEMM activation quantization makes last-ulp drift flip int8 rounding — get a cosine gate plus a subtler standard: "prove quantized quality by showing your outputs sit at the same distance from the f32 truth as the reference's own" (`docs/PORTING.md` §4).

The policy's corollaries, "each learned the hard way" (`docs/PORTING.md` §4), sharpen it further:

- **Bit-exactness, where a discrete decision genuinely rides on ulps, is platform-scoped.** Achieving it means reproducing the shipped binary's *actual* arithmetic — its f16 intermediates, its fp contraction, its vendored BLAS, its system libm — and the resulting contract holds only on the platform whose arithmetic you matched. A universal bitwise claim across platforms is almost always an overclaim.
- **For numerically chaotic models, the oracle sets its own bar.** The method: run the reference against *itself* in two configurations and require your implementation to sit inside the reference's own disagreement envelope — "do not chase a bar the oracle itself cannot meet." Fucina's text-diffusion port is the live example: DiffusionGemma's logit parity "sits inside llama.cpp's own cached-vs-unified numeric spread on this model (the model is numerically chaotic; small kernel differences amplify)" (`docs/BENCHMARK.md`).
- **Parity-critical control paths stay verbatim scalar host code** — sampling bookkeeping, top-k selection, log-softmax schedules. "They are never the bottleneck, and optimized kernels' reassociation breaks near-tie decisions." Optimizing the 0.01% of runtime that decides *which token wins a tie* is how you buy a parity bug with no speed to show for it.

### The inventory, oracle by oracle

With those two foundations, here is Fucina's full verification arsenal. Each row is a technique you can transplant:

| What is verified | Against what | Where |
| --- | --- | --- |
| Tokenizers | token-ID-exact vs `llama-tokenize`, adversarial fixtures (unicode classes, emoji ZWJ, whitespace runs, empty input) | `docs/PORTING.md` §5; the `--tokenize` runner mode |
| Model forward passes | logit parity vs llama.cpp's `--save-logits` dumps, from *explicit raw token ids* so tokenizer bugs can't masquerade as model bugs | `examples/qwen3.zig` `--logits-out` / `--compare-logits`; `tools/llama_logits.cpp` |
| Quantization encoders | byte-exact vs ggml, embedded goldens | `src/backend/quant/encode_golden_test.zig` ([Chapter 11](11-model-files-and-quantization.md)) |
| GGUF writer | byte-identical re-emit of a real model file (`cmp`-equal) | `src/gguf_tests.zig` ([Chapter 11](11-model-files-and-quantization.md)) |
| Optimizers (SGD/AdamW/Muon/APOLLO) | golden-parity vs their torch reference implementations | `src/optim.zig`; `docs/ARCHITECTURE.md` *Training And Persistence* |
| Evolution strategies | golden-pinned and cross-checked **bitwise** against the reference implementation | `tools/gen_es_goldens.py`, `tools/check_es_parity.py` ([Chapter 9](09-training-without-gradients.md)) |
| Autograd gradients | finite differences — the calculus definition of the derivative, used as a unit test | `src/ag/gradcheck.zig` |
| Fast kernels | the scalar backend, `-Dbackend=scalar` | `src/backend/parity_test.zig` |
| SIMD dot kernels across ISAs | semantic asserts vs scalar + FNV-1a checksums diffable across environments | `src/x86dot_check.zig` |
| Documentation snippets | compiled and executed against the real modules on every push | `zig build snippet-check` (§16.4) |

Two of these deserve a closer look.

**Gradcheck: mathematics as the referee.** The optimizer goldens verify the *update rule*; but who verifies the gradients themselves? [Chapter 7](07-autograd.md) built the backward engine; `src/ag/gradcheck.zig` carries its referee, exported publicly as `fucina.gradcheck`:

```zig
pub const Options = struct {
    eps: f64 = 1e-3,
    abs_tol: f64 = 1e-3,
    rel_tol: f64 = 1e-2,
    print_mismatch: bool = true,
};

pub fn gradcheck(ctx: *ExecContext, comptime loss_fn: anytype, inputs: anytype, options: Options) !Result {
```

The idea fits in one line of calculus: if autograd claims `∂loss/∂x = g`, then nudging `x` by `±eps` must move the observed loss by about `2·eps·g`. The harness perturbs each element of each variable input, re-runs the loss, forms the central difference, and compares against the analytic gradient within `abs_tol`/`rel_tol`. Here is the principle, minimal and self-contained — course code:

```zig
// Course code — finite differences in miniature.
const std = @import("std");

fn f(x: f64) f64 {
    return x * x * x - 2.0 * x; // f'(x) = 3x^2 - 2
}

fn fPrime(x: f64) f64 {
    return 3.0 * x * x - 2.0;
}

test "gradcheck in one line of calculus: central differences referee the analytic gradient" {
    const eps = 1e-5;
    var x: f64 = -2.0;
    while (x <= 2.0) : (x += 0.25) {
        const numeric = (f(x + eps) - f(x - eps)) / (2.0 * eps);
        const analytic = fPrime(x);
        try std.testing.expectApproxEqAbs(analytic, numeric, 1e-6);
    }
}
```

What makes the real thing production-grade rather than a toy is the contract in its doc comment (`src/ag/gradcheck.zig:1-12`): the loss must be deterministic, inputs must be contiguous "because the harness perturbs their owned storage directly", constants may appear in the input tuple but are ignored. The oracle arrives *with* its preconditions stated — an oracle with unstated assumptions is a future debugging session. Per `README.md` (*Training*), Fucina's gradients were verified three independent ways: PyTorch goldens, finite differences, and a real-model audit. Independent oracles are how you catch the case where two implementations share a wrong assumption.

**Runners double as oracle harnesses.** There is no separate "parity tool" directory that rots apart from the code it checks. Every model runner *is* the parity harness for its family: the qwen3 CLI exposes `--tokenize` (rung 1 of the ladder below), `--logits-out`/`--compare-logits` (rung 2), and `--verify-cache` alongside its ordinary `--chat`/`--gen`/`--bench` modes (`examples/qwen3.zig`); the multi-stage ports expose exit-code compare modes — parakeet's `--compare <stage> <dump>` flag, omnivoice's `compare` dump-diff subcommand (`examples/parakeet.zig`, `examples/omnivoice.zig`). The tool users run and the tool that proves correctness are the same binary, so the oracle cannot drift from the product.

And one meta-rule guards them all — `docs/PORTING.md` §2: **"Verify your verifiers. A harness that prints PASS/FAIL is not a gate until it exits nonzero on failure."** The same passage names the two classic ways a verifier fools its author: "accept criteria must be behavioral (a name-grep passes on the wrong code); structural checkers pass on code that doesn't compile." A green checkmark that a script cannot mechanically consume — or that would stay green on broken code — is decoration. Every gate in this chapter is an exit code, and the cheapest way to trust a new gate is to break the thing it guards, once, on purpose, and watch it go red.

### Determinism is a contract, not a mood

Oracles presuppose reproducibility: you cannot compare two runs that were never going to agree with themselves. That is why determinism in Fucina is not a nice-to-have but a listed invariant with teeth (`docs/DEVELOPMENT.md` §1.9). Three commitments, escalating in subtlety:

1. **The RNG's (seed → values) mapping is a checkpoint contract.** [Chapter 5](05-the-operation-library.md) introduced the repo-owned counter-based RNG (`src/rng.zig`); here is the *why* at full strength. Dropout masks, APOLLO's random projections, and ES noise ([Chapter 9](09-training-without-gradients.md)) are **regenerated from seeds, not serialized** — a training checkpoint stores a seed and trusts that replaying it reproduces the exact noise. That works only if the mapping from seed to values never changes. Hence the two prohibitions: "Never swap in `std.Random` for anything seed-persisted; never change the fill algorithms without accepting that existing checkpoints break." The RNG is part of the on-disk format.
2. **Parallel kernels are bitwise-deterministic for any thread count** — or they document their rounding class precisely. A reduction that partitions differently across 4 and 8 workers sums in a different order and drifts; Fucina's kernels either fix the reduction tree so the answer is thread-count-independent, or (for the opt-in cases) say exactly what varies.
3. **Even data pipelines pin their randomness**: the SFT loader's `(seed, epoch) → permutation` mapping is golden-pinned, because a shuffled dataset order that silently changes between versions makes training runs incomparable.

> **ML note** — Determinism is undervalued until the day you need to bisect a training divergence. If run N and run N+1 differ only by your change — same weights, same data order, same dropout masks, same ES perturbations — then a difference in the loss curve *is* your change. Without seeded determinism, every comparison is statistical, every bisection needs replicates, and every "it got worse" is an argument instead of a diff. `docs/PORTING.md` §4 states the flip side honestly: cross-*implementation* float equality is corpus-dependent and never guaranteed — "The only universal guarantee is same-build determinism." Fucina makes sure it actually holds.

### The ratchet

Oracles would be pointless if passing them once were the end. The gates are *frozen as a non-regression ratchet*: every later optimization re-passes the full parity battery, and — `docs/PORTING.md` §7 and §4, the first sentence restated in `docs/DEVELOPMENT.md` §4.3 —

> an optimization that flips a single token is reverted, not tolerance-adjusted. ... The tolerance is never loosened to make a gate pass.

This is the sentence that separates verification-as-culture from verification-as-checkbox. The moment a tolerance is loosened to admit a change, the gate stops being an oracle and becomes a negotiation. Fucina's rule removes the negotiation: scheduling-only changes must prove themselves *bitwise* identical; any change that alters floating-point summation order owes a fresh tolerance argument against the pinned oracles — and if that argument is expensive, the change waits until profiling shows it matters.

## 16.4 Docs are tests: `snippet-check`

Documentation lies in a specific, predictable way: it was true when written, then the API moved. Fucina closes that hole by making the reference manual executable. `zig build snippet-check` — an in-tree build step, run in CI on every push — extracts every runnable ```zig block from `docs/REFERENCE.md` (any fenced block containing a column-0 `test "..."` declaration), generates a test root from them (`tools/gen_snippet_tests.zig`), and runs it against the real `fucina`/`fucina_llm` modules with the build's option set (`docs/REFERENCE.md` §2.7).

The authoring contract is small and worth copying:

- Snippets assume an **implicit prelude** (`std`, `fucina`, `llm = @import("fucina_llm")`, `optim = fucina.optim`) so each block stays short without becoming uncompilable.
- `<!-- snippet: helper -->` marks a definition block (a struct or fn the prose introduces) that gets prepended to every later snippet in the same chapter — prose can build up context the way a tutorial naturally does, and the machine reassembles it.
- `<!-- snippet: skip -->` exists for the rare block that genuinely cannot run hermetically — an explicit, greppable admission rather than a silent one.
- Snippets for **opt-in build features stay runnable, not skip-marked**: a constrained-decoding snippet opens with `if (!llm.llguidance.enabled) return error.SkipZigTest;`, so it *compiles* under every flag combination and *executes* exactly when the feature is enabled.

This is why the course has told you, chapter after chapter, that REFERENCE.md snippets are machine-verified — it is not a figure of speech. When the public API changes, `snippet-check` fails until the documentation is updated, which makes updating the docs part of the change rather than a follow-up ticket that never happens ("Docs are part of the change", `docs/DEVELOPMENT.md` §4.4).

Two smaller pieces of documentation discipline orbit the same idea. `zig build doc-check` (`tools/check_doc_links.zig`) fails when the doc index in `AGENTS.md` names a document that does not exist — a five-minute tool that ends the era of dead links in the map. And the *style* contract for the docs themselves (`docs/DEVELOPMENT.md` §4.4): "Write docs as timeless reference — what exists and how it behaves; no dates (benchmark snapshots excepted), no development narrative." Reference documentation describes the present tree; history lives elsewhere. The one deliberate exception proves the rule twice over: benchmark records *must* carry dates (§16.6), and design records evolve by "dated outcome addenda rather than rewrites" (`docs/PORTING.md` §9) — so a reader can always tell what was believed when, and what superseded it, without the document lying about either.

## 16.5 Tests that scale with the tree, and the CI matrix

The unit-test layer underneath all of this has three structural conventions (`docs/REFERENCE.md` §2.7):

**Sibling test files.** Behavior tests live in `<name>_tests.zig` next to the production file — 143 of them across `src/` and `examples/`. The production file pulls its sibling in with a one-line forwarding stanza:

```zig
test {
    _ = @import("exec_tests.zig");
}
```

so analyzing the production file analyzes its tests, while the production code itself stays uncluttered. Module roots forward everything (`src/fucina.zig` ends in a `test` block referencing every submodule), so a single `addTest` per root reaches the whole tree. `zig build test` runs **nine test roots** — the core, the LLM stack, and seven example programs — each compiled with the same option set as the corresponding executable.

> **Zig note** — `test { _ = @import("..."); }` works because Zig's test runner collects `test` declarations from every file *reachable* from the test root; the `_ =` discard exists purely to make the import reachable. Two documented traps follow. First, the silent-test trap: a new `src/ag/` submodule must be referenced from `ag.zig`'s test block "or its sibling tests silently never run" (`docs/DEVELOPMENT.md` §4.4) — an unreferenced test file is not a failure, it is *nothing*. Second, remember from §16.2 that this stanza forms a benign 2-cycle with the sibling — which is exactly why `arch-check` had to be test-aware.

**Suites skip, never fail, when assets are missing.** All nine roots pass with no model files present. Anything needing external material converts `error.FileNotFound` into `error.SkipZigTest`, parity suites gate on environment variables (`OMNIVOICE_PARITY=1`, `NANOCHAT_PARITY=1`), GPU tests skip unless a provider is built *and* a device is present, and feature-gated tests guard on their comptime flag. The result: `zig build test` is green on a fresh clone, and gains coverage — never failures — as assets and flags are added. A test suite that requires setup to pass is a test suite people stop running.

> **Zig note** — `error.SkipZigTest` is a language-level convention: a test that returns it is reported as *skipped*, distinct from both pass and fail. That distinction is what makes the discipline auditable — the run's summary shows exactly how many suites declined to run, so "green" never silently means "green because nothing executed". The idiomatic guard is one line at the top of the test: `if (!llm.llguidance.enabled) return error.SkipZigTest;` for a feature flag, or a `catch |err| switch (err) { error.FileNotFound => return error.SkipZigTest, else => return err }` around an asset open.

**Passing tests are silent.** "Always-passing tests must not print to stderr" (`docs/DEVELOPMENT.md` §4.4) — success-path diagnostics route through an opt-in `testlog` gate. Noise trains people to ignore output; the only acceptable output of a green run is nothing.

On top sits the CI matrix (`docs/REFERENCE.md` §2.8; `.github/workflows/ci.yml`): two OSes — `ubuntu-latest` (x86-64) and `macos-15` (arm64, pinned rather than `-latest`, bumped deliberately) — with `fail-fast: false` so one OS's failure doesn't mask the other's. The steps, in order:

1. `zig build test` — native backend (Accelerate on macOS, no BLAS on Linux, per the `-Dblas` default);
2. `zig build` — every executable compiles;
3. `zig build bench-check` — every bench executable compiles (bench mains are reachable only through their run steps, so nothing else in the build graph exercises them);
4. `zig build arch-check` — the import-graph gate of §16.2;
5. `zig build doc-check` — the doc-index link gate;
6. `zig build snippet-check` — the REFERENCE.md runnable-snippet gate of §16.4;
7. `zig build x86dot-check` — dot-kernel parity on the host ISA plus the compile-only bit-rot legs;
8. `zig build test -Dbackend=scalar` — ubuntu only: the reference backend;
9. `zig build test -Dblas=none` — macOS only: pure-Zig native kernels, complementing the Accelerate run in step 1;
10. `zig build test -Dllguidance=true` + `snippet-check -Dllguidance=true` — ubuntu only (the runner image ships cargo): un-skips the flag-gated constrained-decoding tests and snippets, keeping the extern ABI, the cargo build, and the Rust-staticlib link from bit-rotting behind a green default build.

Study the shape of steps 8–10: the expensive legs run on *one* OS each, chosen so that "between the matrix and the conditional legs, every backend combination that can run on stock CI hardware is covered" without doubling the bill — scalar gets its coverage on Linux, no-BLAS gets its coverage exactly where BLAS is otherwise the default, and the opt-in feature builds where its toolchain happens to exist. Note the honesty in that sentence's qualifier, too: the CUDA provider is covered by a *compile-only* `cuda-check` leg locally (not in CI), and the ISA arms CI hardware cannot execute (AVX-VNNI, AVX512-VNNI, smmla) are covered by compile-only legs plus dated execution attestations — which brings us to a principle important enough for its own paragraph.

**Compilation proves nothing about numerics.** A kernel arm that compiles for an ISA you've never run on may still compute garbage there — and worse, "emulators can execute unsupported instructions silently wrong rather than faulting" (`docs/PORTING.md` §8). Fucina's answer is the execution-attestation table in `src/x86dot_check.zig`: a dated, per-arm record of where each SIMD arm has actually *executed*, kept in the module header where no one can miss it:

```zig
//!   arm                              | executed on                        | attestation
//!   aarch64 sdot asm                 | natively, Apple M1 Max             | ongoing: zig build test + zig build x86dot-check
//!   aarch64 smmla asm (FEAT_I8MM)    | NEVER — M1 lacks I8MM              | compile+objdump-verified 2026-06-11; execution needs Graviton3+/Grace hardware
//!   x86 AVX512-VNNI (EVEX vpdpbusd)  | NEVER                              | compile-verified only; execution needs Ice Lake/Zen4 hardware
```

Read the second and third rows again: the table says **NEVER**, in capitals, about the project's own code. That is the verification religion in one word — the honest state of an arm no local machine can execute is *recorded as unexecuted*, not rounded up to "tested". The checker itself prints FNV-1a checksums of raw result bit patterns so runs from different environments can be diffed for bit-exactness, which is how the x86 arms were eventually attested: the same checksums, bit-equal, from an M1 under Rosetta and a Raptor Lake box running natively.

## 16.6 Honest benchmarking

Correctness has oracles; speed has thermodynamics. CPU benchmarking on laptops is hostile territory — thermal throttling, page-cache state, and background load routinely produce differences larger than the effects being measured — and `docs/BENCHMARK.md` is Fucina's protocol for extracting durable claims from it. The README's performance section compresses the stance into three words — "Measured, not asserted" — and the record itself opens with two ground rules for reading it:

> - **Every number carries its hardware and measurement conditions.** CPU benchmarks on laptops are thermally and page-cache sensitive and shape-specific; a number without its conditions is not a result.
> - **Losses are recorded as plainly as wins.** Where llama.cpp is faster, this file says so, with the measured ratio.

And a framing constraint the whole record lives under: it "is one snapshot, taken as of 2026-07-04" (llama.cpp build 30af6e2, every reference pinned to its exact commit in `tools/fetch_refs.sh`), and — from the README's status section — "Benchmarks age." Every number in this section inherits those conditions.

### The paired gate

`tools/bench_gate.py` is the tool behind any "parity-or-faster" claim, and it is deliberately conservative (`docs/BENCHMARK.md`, *The paired benchmark gate*):

- every row runs in **both process orders** (Fucina→llama, then llama→Fucina), so order effects and thermal drift show up in the samples;
- rows whose cross-sample coefficient of variation exceeds `--max-cv` (default 8%) are reported **NOISY**, not counted as results — cool down and rerun;
- raw stdout/stderr and exact command lines are saved for every subprocess, so any row can be audited later;
- it compares **median** tok/s and exits nonzero when Fucina is below `--min-ratio` (default 1.0).

Same machine, same GGUF file, same thread count, CPU-only on both sides, one process at a time, page cache prewarmed for 20+ GB models. The whole protocol is one reproducible invocation (`docs/BENCHMARK.md`):

```sh
zig build -Doptimize=ReleaseFast
python3 tools/bench_gate.py \
  --models qwen3-0.6b-q6_k \
  --tasks prefill,decode \
  --lengths 1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,64,127,128,129,256 \
  --rounds 1 --fucina-reps 3 --llama-reps 3 --cooldown-s 30
```

Every design choice answers a way benchmarks lie: paired orders answer drift, the CV cutoff answers noise dressed as signal, medians answer outliers, saved raw output answers "trust me", and the nonzero exit code turns the whole thing into a gate a script can enforce. Note also what is *not* a knob: the reference binary is llama.cpp built stock from its pinned commit — "Benchmarks always run stock pinned references — never benchmark a patched reference" — and built on the benchmarking machine with no `-Dtarget`, because "a cross-built baseline binary benchmarks the wrong kernels." The standing fairness note is recorded too — and it cuts *against* the home team: Fucina's benchmark loops still perform final logits/sampler work that `llama-bench` skips, so passing rows are conservative for Fucina and failing rows deserve a no-logits A/B before being called kernel regressions.

Even the benchmark *matrix* encodes a lesson. Prompt length is a first-class parameter because "CPU kernels have tile sizes ...; models do not": the routine matrix `1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,64,127,128,129,256` deliberately includes tail-only lengths, one-exact-tile, tile-plus-tail, and boundary cells, because a kernel that shines at pp128 can limp at pp33 — and quoting `pp4` alone is how you fool yourself ([Chapter 6](06-going-fast-on-cpus.md) showed the tile mechanics; `docs/BENCHMARK.md`, *Why prompt length matters*).

One more protocol principle stitches the two regression tracks together: **"Keep correctness separate from throughput."** Throughput runs may use synthetic token ids — the values are fixtures, the length is the variable — while correctness is its own paired check with its own tooling: llama.cpp's debug binary dumps per-position logits for an explicit token-id prompt, and the Fucina runner compares against them (`docs/BENCHMARK.md`, *Correctness check*):

```sh
refs/llama.cpp/build-cpu/bin/llama-debug \
  -m models/Qwen3-0.6B-Q6_K.gguf \
  -p ids:9707,847,829,374 \
  -ngl 0 -t 8 -tb 8 -b 4 -ub 4 \
  --save-logits --logits-output-dir compare/llama-q6-f32kv

zig build -Doptimize=ReleaseFast qwen3 -- \
  models/Qwen3-0.6B-Q6_K.gguf \
  9707,847,829,374 \
  --compare-logits compare/llama-q6-f32kv/llamacpp-Qwen3-0.6B-Q6_K.bin
```

The expected signal is stated with the tolerance policy's precision: for quantized formats, "top-token alignment plus bounded logit drift. Exact bit equality is not expected." A speed number and a correctness claim are different measurements with different protocols, and a benchmark that conflates them — "it's fast and the output looked fine" — establishes neither.

### What losses look like, kept on the record

The cultural core of the protocol is that its scoreboard prints defeats. From the 2026-07-04 snapshot: on Apple M1 Max (8 threads, macOS, llama.cpp with its Accelerate backend), of 236 paired sweep cells Fucina is faster in 221 and at parity in 13 — with two cells on llama.cpp's side, both itemized: Qwen3.5-0.8B Q8_0 pp32 at **0.86x** ("The loss is confined to this shape — pp128 1.09x, pp512 1.17x, and decode 1.37x all win"; the bimodal-process root cause is recorded as an open item), and 30B Q6_K decode at **0.88x**, annotated "recorded without page-cache prewarming ... so this cell is likely conditions-bound, but it stands until re-measured." On the Intel i9-13950HX (Raptor Lake, AVX2+AVX-VNNI, Linux, no BLAS on either side — *verified*, not assumed: the record documents the llama.cpp build flags that were checked), Fucina wins all dense quantized formats with paired-gate medians 1.32–1.95x per format, while MoE decode stays at 0.90–0.95x behind, plainly labeled weight-bandwidth-bound.

Notice what "it stands until re-measured" does: a suspicious loss is neither erased nor explained away — it is recorded *with its suspected cause and the condition under which it may be revised*. That is what makes the 221 wins believable.

Losses also earn a place because they teach. The M1 dense tables all show a llama.cpp discontinuity at pp32 — annotated in the record as llama.cpp's switch to its Accelerate/AMX BLAS path at batch ≥ 32, which "only re-amortizes at larger batches". Knowing your reference's *phase changes* is part of benchmarking it fairly: a sweep that only sampled pp32–64 would flatter Fucina; one that only sampled pp256 would hide the transition entirely. The routine matrix samples both sides of every threshold — the opponent's as well as your own.

### A loss, investigated

The record's most instructive entry is a defeat that became a case study. In the first x86 paired-gate matrix (i9-13950HX, Raptor Lake, Linux, no BLAS either side, llama.cpp build 30af6e2, snapshot 2026-07-04), Qwen3-30B MoE mid-batch prefill lost *decisively*: the pp15–33 band measured 0.36–0.52x. The record did three things a less disciplined project would not:

1. **It recorded the loss with a mechanism hypothesis**, not just a number: at 128 experts routed top-8, pp8–33 gives each selected expert only ~1–3 rows of work, so "each expert's full weight matrix streams from memory for almost no work — a weight-bandwidth shape llama.cpp's `mul_mat_id` handles better."
2. **When the fix came (update dated 2026-07-10), it ran a reverted-gate control first**: before claiming the new scheduler recovered the band, the old code path was re-measured and reproduced the original 0.375–0.509 ratios — proving the llama.cpp side of the comparison was still directly comparable, and the improvement wasn't a change in conditions wearing a fix's clothes.
3. **It decomposed the fix into its two levers and measured each separately** — the phased-chain gate lowered from 512 to 64 routed pairs did the heavy lifting; small-m column chunking added the last few percent that took Q6_K over parity at pp15–17 — and then it recorded the *residual*: Q6_K pp31–33 still sits at 0.965–0.987x, "open, small", attributed with evidence (a knob sweep that moved nothing) to the packed layout's byte expansion rather than scheduling.

Loss → hypothesis → control → decomposition → residual, every step with conditions attached. The diagnosis ("scheduling, not kernels") was only possible because the loss had been recorded precisely enough to falsify explanations against. A benchmark record that buries its defeats loses exactly this: the trail that turns a bad number into an engineering result.

The record's *Thermal discipline* section is a confession that doubles as a manual: "The single largest source of wrong conclusions in this file's history was chip temperature." Heat soak inverts thread scaling; long sweeps depress late rows; several apparent "llama.cpp wins decode" readings "evaporated when re-measured cool and prewarmed"; one recorded bad run traced to an accidentally-Debug binary plus swap pressure. Fucina's authoritative comparisons are cool, isolated, best-of A/B pairs with pre-cooldowns — and the file says so, so a reader re-running the numbers knows what conditions the claims assume.

### The gates around the benchmarks

Two more pieces complete the speed track. `tools/opbench_gate.py` covers what a llama.cpp comparison cannot — dispatch latency, autograd tracking overhead, training forward/backward throughput have no llama.cpp counterpart — by gating against a locally recorded per-machine baseline:

```sh
python3 tools/opbench_gate.py record          # once per machine/toolchain
python3 tools/opbench_gate.py check           # nonzero exit on regression
```

Its three rules are a study in matching the gate to the noise character of what it measures: **timings** gate on the median of N repeats within `--tol` (default 10%), report NOISY above 12% CV, and every exceedance is confirmed by a re-run after a cooldown, because transient background load must not fail the gate but a real kernel regression reproduces; **allocation counts gate exactly**, because they are deterministic — any increase fails, and any *decrease* asks for a re-record rather than a silent pass; and **checksums gate exactly**, because "a checksum change is numerical drift, never noise." Baselines are keyed to hostname, arch, and Zig version, and `check` refuses a mismatched environment without `--force` — the gate knows its own numbers are machine-specific and refuses to compare apples to oranges by default.

And `zig build bench-check` compiles every bench executable without running one, because bench mains are reachable only through their run steps and otherwise rot silently — "five of them did exactly that before this step existed" (`docs/BENCHMARK.md`). Even the meta-infrastructure gets a gate, and even the gate's origin story is recorded as a failure count.

## 16.7 The porting method: how a model family earns its parity claim

Model family after model family — Qwen3, Gemma 4, Qwen3.5, DiffusionGemma, Parakeet, OmniVoice, NAM — arrived by the same method, and `docs/PORTING.md` is that method "written for the next port". It is the verification religion applied to a specific, recurring job: reimplementing someone else's neural network and proving you got it right. The skeleton:

**1. Pin the reference before writing any code.** The reference implementation is cloned at an exact commit (`tools/fetch_refs.sh` holds every pin), built locally, and one fixed input's output recorded as the hard target. The contract is written "from the actual artifact, not the paper": parse the real weights file and verify every geometry assumption against its metadata, because "papers, model cards, and your own task notes will be wrong about layer counts, kernel sizes, padding modes, and tensor names." The numeric traps get enumerated up front as a ranked checklist — tie-break directions, accumulation order and width, epsilon placement, `>` vs `>=`, RNG draws per step, and every approximated primitive the reference ships. "Parity means reproducing these idiosyncrasies, not the textbook function."

**2. Build the oracle before the port.** Mirror the reference's own dump surface rather than inventing a comparison format; if it has none, instrument it minimally with env-gated hooks kept as a re-applyable patch — and verify the instrumented binary still produces byte-identical final output, because "dumps must not perturb the thing they measure." Dumps are self-describing (magic + dtype + dims header, layout recorded explicitly — "layout confusion is a classic silent-parity killer"). Deterministic mode is the first oracle; only after greedy passes do you port the RNG exactly, draw-count and all.

**3. Stage the port behind mechanical gates.** Pipeline-ordered stages, one exit-code parity gate each, exposed as `--compare <stage> <dump>` modes in the port's own CLI. Intermediate checkpoints exist for one purpose: to localize the *first divergent stage*. The final gate is exact match of the discrete output. Stage *order* is itself a tool: "port the codec *decoder* before the *encoder*" — because the reference encoder produces the oracle inputs your decoder consumes — and "close an end-to-end exact-output loop as early as possible via the simplest head, so every later change has a full-pipeline oracle." And the rule that keeps the whole method honest: **"never fabricate parity**: if a fixture is missing, the honest terminal state is an explicit BLOCKED naming what a human must supply — not an invented number."

**4. For LLMs, climb the ladder in fixed order** (§16.3 already used its rungs): tokenizer token-ID-exact → logits from explicit raw token ids → only then trust generation, acceptance rates, and benchmarks. "Each rung isolates the next from contamination" — a tokenizer off by one token would otherwise surface as a mysterious logit mismatch three layers away. Loader discipline rides along: resolve every expected tensor by name with shape and dtype-class checks, "and assert the resolved count equals the file's total tensor count" — total coverage surfaces variant drift immediately.

**5. Prove interop in both directions.** Import parity is half a port; the other half is that *upstream accepts your exports* — llama.cpp serving a Fucina-merged GGUF ([Chapter 15](15-training-llms-on-cpu.md) closed exactly that loop), upstream's NAM engine loading a Fucina-trained profile ([Chapter 10](10-the-guitar-amp.md)). Practices: tolerant reader, canonical writer; validate the writer against the *strictest* consumer in the ecosystem, not the mainstream one; when a container cannot express the source format's structure, embed the original file byte-verbatim next to the derived tensors and `cmp` the round trip (the GGUF writer's metadata passthrough from [Chapter 11](11-model-files-and-quantization.md) is the same pattern); and keep a **numbered deviations register** where every intentional divergence from upstream carries its justification and measured magnitude, so later parity audits can classify findings against it instead of re-litigating them — including the case where upstream's own writer and reader disagree, which gets recorded as an upstream bug so it isn't mistaken for yours.

**6. Performance only after parity — with the gates as the ratchet.** And one warning from §7 that completes the picture, because parity has a blind spot: "parity tests pass even when a fast kernel arm is missing and dispatch falls to a scalar path." Correctness oracles cannot see *which code ran*. **The profiler is the completeness oracle parity cannot be** — after wiring a new format, you profile-confirm the hot path is armed, or you ship something correct and quietly 10x slow. Wrong and slow, the two regression tracks, each needing its own instrument: that is the chapter's opening sentence again, discovered independently by the porting method.

**7. A new ISA is a port too.** `docs/PORTING.md` §8 applies the same method to porting *Fucina itself* to a new architecture, and it is the purest illustration of oracle-first thinking. Phase 0 is deliberately correct-but-slow: comptime-guarded portable bodies for every arch-specific primitive so the whole engine compiles, then the *existing* parity suite runs on the new target before any fast kernel is written — "the oracle is ISA-agnostic and travels." Before kernel work starts, a three-tier portability map sorts every primitive: portable as-is / principle transfers but must be re-expressed / architecture-specific with named instruction analogs — with the sober warning that "a compiling fallback in tier two does not deliver the win, and every tuned constant is a host fact to re-derive." The scalar backend of §16.3 is what makes this possible at all: because the specification is executable and portable, correctness on a new ISA is a test run, not a research project, and only *speed* remains to be earned arm by arm — each arm then entering the attestation table of §16.5 as compile-verified, emulated, or hardware-executed, dated.

The method's last section (§9) closes the loop on honesty: "Inference parity does not certify training parity — the training pipeline is a separate parity target with its own audit, and the audit should expect the docs to overclaim: correcting them is part of closing it." Claims name their pipeline — inference, training, or export — because a true statement about one is an overclaim about the others.

## 16.8 Decisions with receipts

A codebase accumulates not only code but *decisions*, and undocumented decisions get re-made — badly — by the next person. Fucina's discipline is that significant design adjudications are written down with their evidence, and rejected alternatives are recorded with measurements so they stay rejected for reasons rather than by inertia.

The flagship example is `docs/MEMORY-MODEL.md` §4, "Why an arena was rejected". [Chapter 3](03-tensors-from-scratch.md) covered the buffer pool; what the design record adds is the *adjudication*: a `std.heap.ArenaAllocator` per forward pass — the "obvious" allocation strategy for an ML runtime — fails on four substantive axes, each with file:line evidence: (1) peak memory regresses from working-set to sum-of-all-intermediates, because the pool reclaims a buffer the instant a transient dies while an arena frees nothing until reset — a forward balloons to roughly `n_layer ×` the activation footprint; (2) it destroys cache locality, because bump allocation returns a fresh address per op while the pool returns the *same* address for same-sized successive allocations, keeping the hot buffer warm in L1/L2 — and that address reuse is asserted by a test; (3) it cannot express refcounted views and KV-cache aliasing; (4) it is impossible for training, where activations must outlive the forward. The rejection is binding: `docs/DEVELOPMENT.md` §1.4 says "do not reintroduce one."

The same pattern recurs at smaller scale throughout `docs/BENCHMARK.md`'s *Recorded negatives*: residual-add epilogues "measured and declined" with the profile counters that showed the whole opportunity under 1.2% of forward time; a `trans_ab` contraction kind declined because every production call site was verified to resolve to other forms; a Q5_K decode repack "tried and reverted" because it regressed prefill; q8_0 KV cache recorded as "a capacity option, not a speed option" with the measured reason. Entries record their re-open triggers — the norm is "do not re-try without new evidence" (`docs/DEVELOPMENT.md` §5), with the strongest entries naming exactly what that evidence would be. `docs/DEVELOPMENT.md` §5 elevates this to a principle: **"Negatives are results."** A tried-and-reverted lever with its measured reason is knowledge; a silently abandoned branch is a trap for the next contributor. And symmetrically: "re-verify any recorded premise against the current tree before building on it" — stale premises in old notes have nearly shipped regressions.

Deliberate *divergences* get the same treatment as rejections. Fucina refuses zero-size and zero-rank raw tensors — every dimension is ≥ 1 — and `docs/ARCHITECTURE.md` (*Tensor And Storage Model*) documents it as "a deliberate torch divergence, not a gap": emptiness fails loud at the construction boundary instead of surfacing as torch's empty-reduction contract (`mean` → NaN, `min`/`max` → runtime error) "deep in a graph"; data-dependent cardinality lives host-side, where Zig represents and guards it natively; and op/backend contracts stay parity-pinned only over non-degenerate shapes, keeping that surface small for every current and future backend. Whether or not you agree with the call, notice its form: the alternative is named, the failure mode it avoids is named, and the downstream simplification it buys is named. A divergence documented like this can be revisited on its merits; an undocumented one just looks like a missing feature.

> **Zig note** — Some decisions are enforceable in the language rather than in prose, and Fucina prefers it that way whenever possible. §16.1's `@compileError` guard is one form. Another is the exhaustive-`switch` policy (`docs/DEVELOPMENT.md` §1.7): dispatch over dtypes and backends deliberately avoids `else` arms, so adding a variant *forces* edits at every dispatch site — "the compile error is the enforcement. A silent `else` that swallows a new dtype defeats the design." The hierarchy is: compiler-enforced beats machine-checked beats review-checked beats documented — but every rule lives at the *highest* level it can, and the ones stuck at the bottom are labeled as such.

## 16.9 The delivery loop: how a change ships

`docs/DEVELOPMENT.md` — "the connective tissue", in its own words — assembles the gates of this chapter into the working method a change actually flows through. Its one-line version:

> find the existing capability first, extend it at its designed seam, verify against the reference backend and the gates, measure before claiming speed, and report what you actually did.

Four stations, each with a lesson worth exporting.

**Check before you build.** The document's own diagnosis: "The most common failure mode of a capable contributor on this codebase is rebuilding something that exists." So §2 is a lookup table from need to existing capability — *you need a contraction* → `einsum` is THE contraction engine, don't hand-roll permute+matmul chains; *you need a new pointwise op with autograd* → `elementalUnary`/`elementalBinary` give you a SIMD-chunked parallel op with a VJP from two scalar functions; *you need a linear layer in a model* → `LinearWeight`, whose dispatch to BLAS/Metal/CUDA/quant kernels is *inside* — "never hand-roll a linear". And if a capability is genuinely missing, the design records come *before* the design: several "obvious" additions — arenas, LUT kernels, negative-step views — "were evaluated and rejected with measurements, and the records say why" (§16.8's registers, working as intended). §3 continues with a template table: every kind of new work has a named best-in-class precedent in the tree — a llama-shaped LLM starts from `src/llm/qwen3/`, a pure-CNN vision port from `examples/facedetect/`, a training pipeline from `examples/nanochat/` — because the cheapest design review is reading the thing that already survived one.

**Plan with mechanical accepts.** Work is structured as items with three parts — *Do* (imperative, with code anchors), *Accept* (a mechanical gate: an exit code, a named test, a grep that must return nothing), *Refs* — and an item is ticked only when its Accept gate and the full build pass (`docs/DEVELOPMENT.md` §4.1, generalizing `docs/PORTING.md` §3). "Done when" is stated *before* starting, including the honest negative arm: "works per-stream OR documents why it stays single-stream" is a legitimate completion. The point of an Accept gate is that it cannot be argued with after the fact — you either have the exit code or you don't.

**Run the gates your change can affect.** The full menu, matched to blast radius (`docs/DEVELOPMENT.md` §4.2):

| Gate | What it proves | Run when |
| --- | --- | --- |
| `zig build test` | nine test roots, native backend, no assets needed | always |
| `zig build test -Dbackend=scalar` | native agrees with the reference backend | anything numeric — once, on final code |
| `zig build test -Dblas=none` | pure-Zig kernels unbroken | anything numeric near GEMM dispatch |
| `zig build arch-check` | layering intact (zero SCCs) | new files / imports |
| `zig build doc-check` | doc index resolves | doc adds/moves |
| `zig build snippet-check` | every runnable REFERENCE.md snippet compiles and passes | any REFERENCE.md edit; any public-API change |
| `zig build x86dot-check` | cross-ISA int8/quant dot parity + compile-only ISA legs | quant kernel / dot-arm changes |
| `zig build bench-check` | every bench main still compiles | bench/ or op-signature changes |
| Family parity oracles (`--tokenize`, logit parity, `--compare`) | model behavior unchanged | anything touching a family |
| `tools/bench_gate.py` / `tools/opbench_gate.py` | speed did not regress | any kernel/perf/hot-path change |

Two footnotes to the table encode hard-won economics. The scalar leg runs *once, on final code* — it is slow by design, and running it per-iteration is a tax that teaches people to skip it entirely; the practical cadence is native suite while iterating, scalar before merge. And anything under `src/backend/`, `src/exec/`, or a family's forward path needs **both** tracks: "correct-but-slower and fast-but-wrong are both regressions, and neither is accepted alone" — with one carve-out, stated rather than smuggled: a speed cost that is the unavoidable price of a real correctness fix is acceptable *if declared explicitly with before/after numbers*, "so the trade is a recorded decision rather than a surprise" (`CONTRIBUTING.md`).

**Report what you did.** The closing station, from `CONTRIBUTING.md` and `docs/DEVELOPMENT.md` §4.5: the PR includes the exact commands run, the machine and backend configuration (CPU, OS, threads, `-Dblas`/`-Dbackend` flags), the model and quantization used, and any failures or skipped suites — because **"'Tests pass' without the machine and the commands is not a report."** A report is evidence someone else could re-run; anything less is a mood. It is the same standard the benchmark record holds itself to (§16.6), applied to every change: claims carry their conditions.

The loop closes with three habits `docs/DEVELOPMENT.md` §5 calls "honest completion" — the culture the mechanics exist to serve:

- **Claims name their pipeline.** Inference parity does not certify training parity; export interop is its own gate. Say which of the three a result covers.
- **Negatives are results.** A tried-and-reverted lever, recorded with its measured reason, is a valid completion of a plan item.
- **BLOCKED beats fabricated.** "Missing fixture, missing hardware, missing reference — say so and name what a human must supply. The parity method only works because nobody invents its numbers."

That last sentence is the whole chapter in one line. Every oracle, gate, and protocol described here is downstream of a single social fact: the numbers are real, all of them, including the missing ones.

## 16.10 Growth through applications

Look back at the course's arc and a pattern emerges: every major capability arrived because an *application* demanded it. Conv kernels and streaming state came with the guitar amp ([Chapter 10](10-the-guitar-amp.md)); GGUF, quantized matmul, and the KV cache came with the LLM stack (Chapters [11](11-model-files-and-quantization.md)–[12](12-a-transformer-from-scratch.md)); the MoE scheduler and expert streaming came with the 30B-and-up models ([Chapter 13](13-inference-tricks.md)). That is not an accident of narration. It is the project's stated growth mechanism — `README.md`:

> These applications live in `examples/` and each will eventually graduate into its own repository. Meanwhile they are here for convenience, and not by accident: with the Tensor core in place, Fucina grows and gets tested through real applications, so the runtime and the things built on it develop side by side.

Three examples show what each application *proves* about the core:

- **`examples/nanochat.zig`** — Karpathy's nanochat ported whole: BPE tokenizer training, GPT pretraining, SFT, bits-per-byte eval, chat, parity-targeted against "the Python reference (nanochat @ 92d63d4) running on CPU in fp32" (its module header). Its structural significance is a single fact from `docs/DEVELOPMENT.md` §1.8: nanochat is "entirely example-local over the public facade" — a complete training pipeline needed *zero* new core ops. That is the strongest existence proof a library API can have: the public surface is sufficient for a real system its designers did not anticipate op-by-op.
- **`examples/facedetect.zig`** — a port of mudler/face-detect.cpp: SCRFD detection, ArcFace recognition, gender/age, anti-spoofing, dense landmarks. It proves the runtime is not LLM-shaped — convolutional vision models, multiple networks in one GGUF — and that the parity method transfers to a completely different domain (its goldens are byte-identical JSON outputs).
- **`examples/lmserve.zig`** — an OpenAI-compatible HTTP server: Chat Completions and Responses, SSE streaming, JSON-schema/regex/Lark constrained output, "a bounded request queue in front of one sequential inference worker" (its header). It proves the stack composes into a product surface: any OpenAI client can talk to a Zig binary, with real serving concerns — auth, backpressure with 429s, KV-cache reuse — layered on the same runtime.

> **ML note** — Why is "grow through applications" the right test strategy for an ML runtime in particular? Because the space of ops is effectively unbounded — no unit-test suite enumerates it — but the space of ops *real models use* is concrete and finite. Every port arrives with an implicit specification (its reference implementation) and an implicit coverage set (every op on its forward path, at real shapes and dtypes). A runtime that has survived a WaveNet, several transformer families, a conformer ASR stack, a MaskGIT TTS, and a CNN face pipeline has been integration-tested against the actual distribution of workloads, not a synthetic one — and each survival is certified by a parity oracle rather than by "it didn't crash".

The applications are the test suite that unit tests cannot be: each one forced new ops, new dtypes, new scheduling into the core — *each behind a parity oracle* — and what one application forces in stays for everyone (the `moe_chain` scheduler that Gemma's MoE forced is the same one Qwen's uses, "so scheduler fixes land once for every family", `docs/ARCHITECTURE.md`). The oracles travel with the domains, and the strength of the resulting claims is worth pausing on, each from the dated `docs/BENCHMARK.md` record (M1 Max unless noted): Parakeet ASR transcripts are **byte-identical** to parakeet.cpp in every measured mode, so its speed rows are pure-speed comparisons; OmniVoice's MaskGIT token streams and RVQ codes are **byte-exact** at F32 with a fixed seed, decoded audio cosine ≥ 0.99999; LocateAnything detections are **byte-identical** JSON (`cmp`-equal) with id-exact token streams (measured 2026-07-07; on the i9 x86 leg the f32 streams stay byte-identical cross-engine while q8_0 drifts within the tolerance policy's expectations — and the record itemizes exactly which coordinates, by how much). Speech, audio synthesis, and vision — three domains, one method, and in each case the parity claim is *stronger* than the tolerance policy demands, because the discrete outputs happened to allow it. When the same discipline produces the same shape of result in every new domain, that is evidence the discipline, not luck, is doing the work.

Where each piece *lives* follows a placement policy (`docs/DEVELOPMENT.md` §1.8) that keeps growth from smearing the layers of §16.1: reusable engines other `src/llm` consumers should import go in `src/llm/<family>/`; single-purpose parity ports and their DSP/IO plumbing stay example-local in `examples/<name>/`; family-specific kernel orchestration lives in the family, never inside the generic exec runtime; shared cross-family scheduling lands once in exec and is re-exported. New core tensor ops belong in the exec/backend bands — "but a good port usually needs none: nanochat is entirely example-local over the public facade."

And the boundary of the whole mechanism is explicit: "Research experiments that lack a reference oracle live on `research/*` branches rather than `main`" (`README.md`) — main is for code that can be *verified*, which is the growth thesis and the verification religion agreeing with each other.

## 16.11 Status honesty as a feature

The last discipline is the one that wraps all the others: saying plainly what the project is not. The README's *Status and scope* section leads with the phrase "Honest expectations" and delivers them:

- **"CPU-first, two ISAs."** Tuned on Apple Silicon (NEON/sdot) and x86-64 (AVX2/AVX-VNNI); the scalar reference backend covers everything else. The Metal offload "accelerates specific GEMM shapes on macOS; it is not a general GPU runtime", and the CUDA sibling plugs into the same seam — an offload seam, not a third backend, with its gaps itemized in `docs/ARCHITECTURE.md` (*Current Production Gaps*): no attention/KV offload, no distributed execution.
- **"The API is not stable. This is a young codebase published in the open, not a 1.0 library. Expect churn."** There is no package manifest — no `build.zig.zon` — and "No stable external API contract or versioning" is listed *first* among the production gaps. Every signature this course has shown you is today's code, not a frozen contract.
- **"Model weights are not included"** — each family carries its own license, noted next to each download link.
- **"Benchmarks age."** Every number is a dated snapshot against a moving target.

The gaps list continues where a brochure would stop: encoder coverage stops at K-quants plus the legacy formats, with the cold formats decode/matmul-only ([Chapter 11](11-model-files-and-quantization.md) met this as `error.EncoderUnavailable` — a verification-policy boundary, since an encoder without a byte-exactness oracle would violate §16.3); no unified session abstraction across LLM families yet; no documented thread-safety contract for sharing tensor handles across threads. And then the architecture document grades its own work (`docs/ARCHITECTURE.md`, *Production Readiness Assessment*):

> Current assessment: **production-oriented core, not production-ready product.**

Why treat this as a *feature* rather than a disclaimer? Because trust is transitive through documentation. A reader who has just seen a project state "0.86x, loss confined to this shape, open item" about its own benchmark, "NEVER" about its own untested SIMD arms, and "not production-ready" about its own maturity has calibrated evidence that the *positive* claims — 221 of 236 cells, byte-identical re-emit, token-ID-exact — mean exactly what they say. Projects earn the right to be believed about their strengths by being precise about their weaknesses. Overclaiming, by contrast, is a debt the docs take out against the code, and `docs/PORTING.md` §9's audit stance ("expect the docs to overclaim: correcting them is part of closing it") treats that debt as a bug class with a repair process.

The same honesty extends to how contributions are judged. `CONTRIBUTING.md` expects contributors to work with coding agents — the README says the project itself is built that way, "with humans leading the ideas, the testing, and the debugging" — and draws the line in one sentence: "Line-by-line authorship is not the bar — accountability is." What is required is a human who understands the change, ran the gates, and can answer for it in review; and a report that names its evidence, because "'Tests pass' without the machine and the commands is not a report."

Provenance is part of the same contract, and it is honesty pointed outward. The README's acknowledgments open with "Fucina exists because others built the road first" and then itemize the debts precisely: which components are direct ports of llama.cpp code (the quantization row encoders, two tokenizers, the Unicode tables, the SIMD `expf`, a vendored Metal kernel), which projects are parity oracles rather than code sources, whose GGUF conversions the ports run, where the tagged-tensor idea came from (ZML). The complete inventory — vendored vs ported vs reference-only, each with its license — lives in `docs/THIRD-PARTY-NOTICES.md`, and `CONTRIBUTING.md` makes the norm enforceable: "Uncredited ports are treated as bugs." A project this dependent on reference implementations for its verification method owes its references the same precision it demands of its own claims — and saying exactly what you took is the same discipline as saying exactly what you measured.

## 16.12 The transferable checklist

Strip away the Fucina specifics and this chapter compresses to a method you can apply to any serious systems project:

1. **Name your regression tracks.** Fucina has two: wrong and slow. Knowing the axes tells you which gates a change needs — and which it doesn't.
2. **Draw the dependency direction and check what you can mechanically.** An SCC check over the import graph is a few hundred lines and catches the rot that diagrams can't. Label everything the machine *can't* check as review-only, explicitly.
3. **Turn design rules into compile errors where the language allows it.** A `comptime` guard or an exhaustive switch outlasts any code-review vigilance.
4. **Build the oracle before the feature.** Reference implementation, boring twin, mathematical identity, golden file — something independent, compared mechanically, exiting nonzero.
5. **Keep a boring twin of every fast path.** The scalar version is the specification; the fast version never escapes its judgment.
6. **Tier your tolerances by what the numbers can actually promise** — exact where outputs are discrete, justified tolerance where floats drift — and never loosen a tolerance to make a gate pass.
7. **Make determinism a contract.** Pin your (seed → values) mappings, make parallel code thread-count-independent or document what varies, and bisections become diffs instead of arguments.
8. **Make the docs executable.** A snippet that runs in CI cannot lie about the API.
9. **Date and condition every measurement.** A number without its machine, protocol, and date is not a result. Record losses with the same prominence as wins.
10. **Plan with mechanical accepts, report with evidence.** "Done" is an exit code stated in advance; a report names the commands, the machine, and the failures — "tests pass" is not a report.
11. **Record negatives and rejections with their evidence** and a re-open trigger — a documented dead end is a contribution.
12. **State what the project is not.** Precision about weaknesses is what makes claims about strengths believable.

None of these requires Fucina's scale. The parity twin and the finite-difference check in this chapter are a screenful of code each; an import-graph gate is an afternoon; "losses recorded as plainly as wins" is a decision, not an infrastructure. The craft is not any single gate — it is the habit of never letting a claim outrun its evidence.

And notice, finally, what the habit buys that no single gate could: *compounding*. The scalar spec made the x86 port a test run; the parity ratchet made every optimization safe to attempt; the dated benchmark record made a 0.36x defeat diagnosable and repairable days later, with a control to prove the repair; the sealed facade made nanochat possible without touching the core. Each discipline was paid for once and has been collecting interest since. That is the real argument for the craft — not virtue, but leverage — and it is the note this course wants to end on. The [epilogue](17-epilogue.md) hands you the forge.

## What you now know

- Fucina's quality apparatus is organized by one sentence: the library regresses by becoming **wrong** or becoming **slow**, and every change is gated against the failure modes it can realistically affect — no more, no less (`CONTRIBUTING.md`).
- The source tree is banded — apps → llm → facade → autograd/training → tagged → exec → backend → tags → tensor → primitives → core — and a band may depend only on bands at or below it (`docs/ARCHITECTURE.md`).
- `zig build arch-check` (in-tree, in CI) enforces **zero nontrivial SCCs** over the production import graph with an AST-based, test-aware checker — currently 105 files, 408 edges, 0 SCCs; band *direction* is review-enforced against the layer table, and `docs/DEVELOPMENT.md` explicitly labels its unenforced invariants review-only.
- The verification arsenal, as a reusable set: scalar backend as executable spec; tiered tolerances (discrete = exact, pure-f32 ≈ 1e-4 max-abs, quantized-downstream = cosine ≥ 0.9999x); token-ID-exact tokenizers; logit parity from raw ids; byte-exact quant encoders; byte-identical GGUF re-emit; golden optimizer tests; bitwise ES cross-checks; finite-difference gradcheck; runners doubling as oracle harnesses; and the ratchet — a change that flips one token is reverted, never tolerance-adjusted.
- Determinism is a listed invariant: the repo-owned RNG's (seed → values) mapping is a checkpoint contract (dropout, APOLLO projections, ES noise regenerate from seeds), parallel kernels are bitwise-deterministic for any thread count or document their rounding class, and even the SFT loader's shuffle is golden-pinned — because "the only universal guarantee is same-build determinism", so Fucina makes sure it holds.
- Docs are tests: `zig build snippet-check` extracts every runnable REFERENCE.md snippet and executes it against the real modules in CI, with an authoring contract (implicit prelude, helper/skip markers, feature snippets runnable behind comptime guards).
- Tests scale by convention: 143 sibling `_tests.zig` files reached through forwarding stanzas, nine test roots green with zero assets (missing material skips via `error.SkipZigTest`, never fails), silent success paths, and a two-OS CI matrix covering every backend combination stock hardware can run — with compile-only legs and dated execution-attestation tables for what it can't, because compilation proves nothing about numerics.
- Honest benchmarking is a protocol: paired both-order same-machine runs, median-of-samples, CV-based noise rejection, prewarming, thermal discipline, dated snapshots (2026-07-04, M1 Max and i9-13950HX), a prompt-length matrix that probes tile boundaries, correctness checked separately from throughput — and losses (0.86x, 0.88x, MoE decode 0.90–0.95x) kept on the scoreboard next to the wins.
- A recorded loss is the start of an investigation, not the end of one: the x86 MoE mid-batch band (0.36–0.52x, i9-13950HX, snapshot 2026-07-04) was root-caused to scheduling and recovered (update dated 2026-07-10) via a reverted-gate control, a two-lever decomposition, and an honestly recorded residual — the loss→hypothesis→control→decomposition→residual pattern.
- Documentation has its own gates and style contract: `doc-check` for the index, snippet-check for the API, "timeless reference" prose with dates reserved for benchmark snapshots and design-record outcome addenda.
- A port earns its parity claim by method: pin the reference at a commit, build the oracle first, stage behind exit-code gates, climb the tokenizer→logits→generation ladder, prove interop in both directions, keep a numbered deviations register, and treat BLOCKED as the honest terminal state when a fixture is missing. The profiler is the completeness oracle parity cannot be.
- Decisions carry receipts: the arena rejection in `docs/MEMORY-MODEL.md` §4 and the recorded-negatives register in `docs/BENCHMARK.md` show how to document rejected alternatives with measurements and re-open triggers.
- A change ships through a delivery loop: check the capability tables before building, start from a named in-tree template, plan items as Do/Accept/Refs with mechanical accept gates, run the gate set matched to the blast radius (scalar leg once, on final code; hot-path changes need both tracks), and report with commands, machine, and configuration — "'Tests pass' without the machine and the commands is not a report."
- The library grows through real applications — "not by accident" — each landing behind a parity oracle; and it states its own limits plainly: young unstable API, no package manifest, CPU-first on two ISAs, encoder coverage bounded by verification policy, self-graded "production-oriented core, not production-ready product".

## Explore the source

- `docs/ARCHITECTURE.md` — the layer table, the enforcement section, and the self-assessment; the single best map of the tree.
- `tools/check_import_graph.zig` — an AST-based, test-aware Tarjan SCC checker in a few hundred lines; a model for building lint tools that understand your idioms.
- `src/fucina.zig:28-42` — the comptime seal on the raw layer: a design rule as a compile error.
- `docs/PORTING.md` — the porting method end to end; read it before porting *anything*, in any language.
- `docs/BENCHMARK.md` — the protocol sections (paired gate, prompt-length rationale, thermal discipline) and the *Recorded negatives*; the scoreboard is a masterclass in conditioned claims.
- `docs/DEVELOPMENT.md` — the invariants with their enforcement status, the check-before-you-build table, and the honest-completion rules ("Negatives are results", "BLOCKED beats fabricated").
- `src/ag/gradcheck.zig` — the finite-difference referee: contract-first doc comment, `comptime` loss validation, tuple inputs.
- `src/x86dot_check.zig:1-40` — the execution-attestation table; what "tested" means when hardware you don't own is involved.
- `docs/REFERENCE.md` §2.7–2.8 — test organization, the snippet-check authoring contract, and the CI matrix.
- `docs/MEMORY-MODEL.md` §4 — the arena rejection: the best worked example of documenting a rejected alternative with measurements.
- `CONTRIBUTING.md` — the whole contribution contract in 77 lines; the two-track framing this chapter is built on.
- `src/backend/parity_test.zig` and `src/backend/quant/encode_golden_test.zig` — what scalar-vs-native parity and byte-exact golden testing look like as real test files.

## Exercises

1. **Read a loss report.** In `docs/BENCHMARK.md`, find the Qwen3.5-0.8B pp32 entry and the 30B Q6_K decode entry. For each, list: the measured ratio, the machine and conditions, what has been ruled out, and what would revise the record. Then write the equivalent three-sentence "honest loss entry" for a slow case in a project of yours.
2. **A parity twin of your own.** Take the `sumVector` course kernel from §16.3 and make it wrong in a way the test *doesn't* catch (hint: what happens if you break only the tail path and test only with `xs.len % 4 == 0`?). Then extend the test to sweep lengths 1..64 so the bug cannot hide. This is why Fucina's benchmark matrix includes pp1..pp9 — the same boundary logic, on the speed track.
3. **A finite-difference referee for a real op.** Using `fucina.gradcheck` (exported from `src/fucina.zig`), write a test that checks the gradient of a two-op composite loss — e.g. `mean(tanh(x)²)` over a `Tensor(.{.d})` — with the default `Options`. Then sabotage a VJP mentally: which of `eps`, `abs_tol`, `rel_tol` would catch a gradient that is exactly 2x too large? A gradient with the wrong *sign* on one element?
4. **An import-graph gate for your project.** Write a small checker (any language) that parses your project's import statements, builds the graph, and fails on cycles. Then add one exclusion rule your codebase's idioms need — the analogue of Fucina's test-awareness — and document, in the tool's header, exactly what it does and does not prove, the way `tools/check_import_graph.zig:1-15` does.
5. **(Hard) A paired benchmark gate in miniature.** Write a script that benchmarks two commands A and B: run each 5 times in both orders (ABBBA/BAAAB or simple alternation), compute per-side medians and coefficients of variation, print NOISY instead of a verdict when CV exceeds 8%, save every raw output to a timestamped directory, and exit nonzero when A's median falls below a `--min-ratio` of B's. Test it on two binaries you *know* differ (e.g. the same program at `-Doptimize=Debug` vs `ReleaseFast` — README documents Debug as 10–50x slower), then on the same binary twice — and observe how often an unpaired, single-order version of your script would have declared a winner between identical contestants.

---

[Previous: Training LLMs on your CPU](15-training-llms-on-cpu.md) ·
[Next: Epilogue — your forge](17-epilogue.md)
