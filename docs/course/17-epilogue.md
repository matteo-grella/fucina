# Chapter 17 — Epilogue: your forge

*Part VI — The craft*

Seventeen chapters ago you opened a repository and a promise: that a modern deep-learning stack — the part that usually hides below Python, behind a framework, inside a C++ build and a driver — could be one program, in one language, that you read top to bottom. This last chapter is short. It looks back once, then hands you the hammer.

Three things remain: to look back, to look forward, and to say thank you.

## 17.1 The journey, in one page

It began with an enum. [Chapter 3](03-tensors-from-scratch.md) built a tensor out of nothing but `src/dtype.zig`, a refcounted flat buffer (`src/storage.zig`), and shape/stride arithmetic (`src/tensor.zig`) — a pointer plus index math, with ownership spelled out in `defer` instead of hidden in a garbage collector.

[Chapter 4](04-axes-with-names.md) moved the axis names into the type system: `Tensor(.{ .batch, .in })` is a different type from `Tensor(.{ .in, .batch })`, a misaligned contraction is a compile error, and the tags compile away entirely (README.md). [Chapter 5](05-the-operation-library.md) grew the op library — validate once, dispatch to small unchecked kernels — and [Chapter 6](06-going-fast-on-cpus.md) made it fast with `@Vector` kernels held to the scalar backend, the executable specification.

Then the values learned to remember where they came from. [Chapter 7](07-autograd.md) built autograd with no tape and no graph object — the live tensors *are* the graph, an idea inherited from spaGO and re-executed with atomic dependency counters on a bounded pool (README.md, Origins). [Chapter 8](08-training.md) turned gradients into learning: the same forward function that infers also trains, once an exec scope adopts its intermediates. [Chapter 9](09-training-without-gradients.md) showed learning without gradients at all.

Then the library met the world. [Chapter 10](10-the-guitar-amp.md) closed the circle the introduction opened — Zig itself was conceived inside a digital audio workstation, and here a WaveNet ran a guitar amp live, from a callback that never allocates.

[Chapters 11](11-model-files-and-quantization.md)–[15](15-training-llms-on-cpu.md) climbed the language-model ladder: GGUF and block quantization, a transformer from embeddings to sampled token, the inference tricks that make CPUs respectable, the ternary frontier, and finally training — LoRA on a quantized GGUF, and karpathy's nanochat pipeline ported whole, a GPT trained from scratch on your own machine. [Chapter 16](16-the-craft.md) collected the discipline that made all of it trustworthy: parity oracles, the scalar twin, honest benchmarks, gates a machine enforces.

Three threads never broke across those chapters, and they were the course's real curriculum: code you can read all the way down, verification before optimization, and performance treated as a feature of your own machine rather than someone else's datacenter.

The same arc, as a review index:

| Part | Chapters | What you built or read | Where it lives |
| --- | --- | --- | --- |
| I — Foundations | [0](00-introduction.md)–[2](02-just-enough-ml.md) | the premise, the language, the math | README.md; `AGENTS.md` |
| II — The tensor core | [3](03-tensors-from-scratch.md)–[6](06-going-fast-on-cpus.md) | dtypes, storage, tensors, tags, ops, SIMD | `src/dtype.zig`, `src/storage.zig`, `src/tensor.zig`, `src/tags.zig`, `src/exec.zig`, `src/backend/` |
| III — Learning | [7](07-autograd.md)–[9](09-training-without-gradients.md) | autograd, optimizers, evolution strategies | `src/ag/`, `src/optim.zig`, `src/es.zig` |
| IV — Sound | [10](10-the-guitar-amp.md) | a real-time neural guitar amp | `examples/nam/` |
| V — Language models | [11](11-model-files-and-quantization.md)–[15](15-training-llms-on-cpu.md) | GGUF, quantization, a transformer, inference tricks, low-bit, CPU training | `src/gguf.zig`, `src/backend/quant/`, `src/llm/`, `examples/` |
| VI — The craft | [16](16-the-craft.md) | the discipline that holds it together | `docs/`, `tools/`, the CI matrix |

From a dtype enum to a live guitar amp and chatting language models. One language. One machine. Nothing you cannot read.

## 17.2 Projects, from easy to hard

The only way to keep any of this is to build with it. Six projects, graded. Each one has a documented path in the repo — none requires permission, only patience.

| # | Project | Grade | Builds on | Your first milestone |
| --- | --- | --- | --- | --- |
| 1 | Run every example | easy | the whole map, hands on | one model answering on your machine |
| 2 | Add a pointwise op, end to end | easy → moderate | [Ch. 5](05-the-operation-library.md)–[7](07-autograd.md) | the elemental prototype, gradchecked |
| 3 | Write a quant decoder | moderate | [Ch. 11](11-model-files-and-quantization.md) | ggml-golden fixtures embedded and failing |
| 4 | Port a small model | hard | everything, at once | the pinned reference dumping stage outputs |
| 5 | Train a NAM profile of your amp | moderate (plus hardware) | [Ch. 8](08-training.md), [10](10-the-guitar-amp.md) | a clean, unclipped capture |
| 6 | Contribute upstream | as hard as you choose | [Ch. 16](16-the-craft.md) | a PR report that names its machine |

Notice the pattern in the milestone column: in every project the first thing you build is the thing that can tell you you are wrong. That is the course's method applied to your own work — the oracle comes first.

**1. Run every example.** Begin where the course began — `zig build spirals` needs no downloads at all — then work through the model zoo: `docs/RUNNING-MODELS.md` has copy-paste commands and verified download links — chat with Qwen3, transcribe with Parakeet, clone a voice with OmniVoice, locate objects from a text prompt, play through an amp profile — and the face-detection walkthrough lives with its example, in `examples/facedetect/README.md`. (Weights are not included in the repo and each family carries its own license; the doc notes the terms next to each download.) Build everything with `-Doptimize=ReleaseFast` — Debug is 10–50× slower (README.md). An afternoon spent here turns fourteen table rows in the README into things you have actually touched.

**2. Add a pointwise op, end to end.** The classic first contribution, and every station of it now has a name you know. Two routes:

- *The short route — no core edits.* `elementalUnary`/`elementalBinary` (`src/ag/tensor.zig:1661`, engine in `src/ag/elemental.zig`): supply scalar forward and backward functions, get a SIMD-chunked, parallel op with a VJP (row in docs/DEVELOPMENT.md §2's "check before you build" table; contract in docs/REFERENCE.md §4.4). This is how you prototype an activation in an evening.
- *The full route — a first-class citizen.* Follow `tanh` through the tree and add your op beside it at each stop:
  - the semantic spec: one enum variant plus one `switch` arm in `src/backend/ops.zig` (`UnaryOp`, `unaryScalar`);
  - the vectorized body: behind `vecUnary` in `src/backend/vector/primitives.zig:155`;
  - the eager entry: `src/exec/elementwise.zig`, plus a one-line `ExecContext` method (`src/exec.zig:1101`);
  - the derivative: `src/ag/backward.zig` (`unaryDerivative`; and `unaryUsesOutput` if it is cheaper in terms of the output, like tanh′ = 1 − t²);
  - the public facade: a one-liner in `src/ag/tensor.zig:1754`.
- *Then verify like the repo verifies:*
  - the derivative with `fucina.gradcheck` (finite differences, `src/ag/gradcheck.zig`);
  - the vector leg against the scalar path, the way `src/backend/vector/primitives_tests.zig:106` does for `elu`/`gelu_erf`;
  - the whole tree with `zig build test -Dbackend=scalar`, so the reference backend agrees with native end to end.

> **Zig note** — The full route is less daunting than it sounds, because the compiler walks you through it. `UnaryOp` is switched exhaustively, with no `else` to swallow your new variant — add the enum case and every site that must learn about it becomes a compile error until it does (docs/DEVELOPMENT.md: "the compile error is the enforcement"). The design reviews your patch before any human does.

**3. Write a quant decoder.** Pick a GGUF block format and implement its dequantization. The repo's home for rarely-used formats is `src/backend/quant/cold.zig`, and the bar is set by `src/backend/quant/cold_tests.zig`: every cold decode format is verified bit-exactly against embedded ggml-golden fixtures (docs/ARCHITECTURE.md, Quantized Matmul Boundary). Note the honest scope the repo itself keeps — the cold formats decode and matmul but do not encode — and hold your work to the same golden-fixture standard: generate reference blocks with ggml, embed them, gate on byte equality.

**4. Port a small model.** This is the project that teaches the most, because it forces every layer of the stack through your hands at once. Do not improvise the method — `docs/PORTING.md` *is* the method, written for the next port, and its one-line version is the whole discipline: you don't optimize what you can't verify, so the oracle is built first, parity closes stage by stage behind mechanical gates, and only then does performance work begin. Pin the reference at an exact commit; choose the smallest variant that exercises every code path you must port (PORTING.md's own example: "the Parakeet port anchored on a 110M hybrid precisely because it has *both* decode heads"); climb the parity ladder (tokenizer token-ID-exact → logits from raw ids → generation); and respect the tolerance tiers — discrete outputs gate on exact equality, and the tolerance is never loosened to make a gate pass.

**5. Train a NAM profile of your own amp.** [Chapter 10](10-the-guitar-amp.md) ran other people's profiles; now capture your own rig. `examples/nam/README.md` walks the whole path — reamp-box wiring, the standardized capture signal, level discipline (the trainer refuses clipped input), then `profile` for one-step capture + train + export, or `train --input in.wav --output reamp.wav --out m.nam` from an existing pair. Profiles are exchanged "with the original NAM tooling in both directions" (examples/nam/README.md). There is something clarifying about a project whose loss you can *hear*.

**6. Contribute upstream.** `CONTRIBUTING.md` is short because the bar is simple, and after this course you can meet it. Two rules matter most.

First, *a human sends the PR*: working with coding agents is expected — the project is built that way — but what is required is a human who understands the change, ran the gates, and can answer for it in review; "line-by-line authorship is not the bar — accountability is."

Second, *the two regression tracks*: Fucina regresses in exactly two ways — it becomes **wrong**, or it becomes **slow** — and your change is tested against the failure modes it can realistically affect, no more, no less. A doc fix needs no benchmark; a kernel change needs both tracks, always, with the machine, commands, model, and quant reported ("'Tests pass' without the machine and the commands is not a report").

Remember the API is explicitly unstable — this is a young codebase published in the open, not a 1.0 library (README.md, Status and scope) — so expect churn, and expect your patch to be measured. If your idea has no reference oracle at all, note the repo's own convention: research experiments without one live on `research/*` branches rather than `main` (README.md).

> **ML note** — "Wrong or slow" is worth carrying beyond this repo. Most ML engineering failures are one of the two, and most ML engineering *process* failures come from testing only one: a refactor benchmarked but never parity-checked, or an optimization parity-checked on a shape where the fast path never ran. The paired discipline — oracle plus profiler, correctness plus speed — is the transferable skill this course was secretly about.

## 17.3 A reading list

The repo documents itself; the README's Documentation table is the index. In course order:

| Read | Because |
| --- | --- |
| `docs/ARCHITECTURE.md` | the actual source layout, layer by layer — the map this course walked |
| `docs/REFERENCE.md` | the full public surface with machine-verified snippets — the contract |
| `docs/RUNNING-MODELS.md` | project 1, ready to paste |
| `docs/MEMORY-MODEL.md` | ownership rules and the buffer-pool-not-arena adjudication — a masterclass in documenting a rejected alternative |
| `docs/TRAINING.md` | autograd to checkpoints, including how the gradients were verified |
| `docs/BENCHMARK.md` | the measurement protocol and dated records, losses included |
| `docs/PORTING.md` | project 4 — how every family here earned its parity claim |
| `docs/LMSERVER.md` | the OpenAI-compatible server: API mapping tables and streaming contracts |
| `docs/SPECULATIVE.md`, `docs/CONSTRAINED-DECODING.md` | design records: how a finished investigation is written down |
| `docs/TERNARY.md`, `docs/PTQTP.md`, `docs/PTQTP-RECIPE.md` | [Chapter 14](14-the-low-bit-frontier.md)'s sources: the low-bit frontier, measurements included |
| `docs/CARTRIDGES.md` | post-training the *context* — a corpus distilled into a reusable KV prefix, weights untouched; the fourth road of [Chapter 15](15-training-llms-on-cpu.md) §15.10's menu |
| `AGENTS.md` | build options, repo map, house rules |

And read the shoulders this project stands on, as its README credits them — "Fucina exists because others built the road first":

- **ggml / llama.cpp** — Georgi Gerganov and the ggml authors; "this project would not exist without their work". Fucina speaks formats ggml defined, several components are direct ports of llama.cpp code, and llama.cpp is the parity oracle and performance yardstick throughout.
- **spaGO** — the Go library where the graph-implicit-in-the-values idea was first explored; Fucina's direct ancestor (README.md, Origins).
- **ZML** — the inspiration for the tagged-tensor approach: axis tags carried in the type, operands aligned by name.
- **NeuralAmpModelerCore / neural-amp-modeler** — Steven Atkinson; the reference for the entire NAM example.
- **nanochat** — karpathy's from-scratch GPT pipeline, ported whole in `examples/nanochat/main.zig` and the existence proof that the public facade can train a language model end to end (README.md, What runs today).

They are not the whole list — the README also credits the ports from mudler's face-detect.cpp/parakeet.cpp/locate-anything.cpp, antirez's ds4, ServeurpersoCom's omnivoice.cpp, the vendored MLX Metal kernel and llguidance engine, and more. The complete inventory — what is vendored, what is ported, what is only a parity reference, and under which license — is `docs/THIRD-PARTY-NOTICES.md`. The repo treats uncredited ports as bugs (CONTRIBUTING.md), and so should you.

## 17.4 The forge

*Fucina* is Italian for forge, and the name was never decoration.

A forge is not a factory: nothing comes off a belt finished. It is a small, hot room where a person with judgment applies heat and pressure to stubborn material, checks the work against a straightedge, and hits it again. Every chapter of this course was that loop. Heat: a real application that needed something the library did not have. Pressure: the benchmark, the profiler, the deadline of a 48 kHz audio callback. The straightedge: an oracle — a reference implementation, a finite-difference check, a scalar twin, a golden byte. And then the next hammer blow.

You now know how the tools are made — not metaphorically, but literally: you have read the dtype enum, the stride arithmetic, the SIMD kernel, the backward node, the GGUF parser, the sampler. The stack is not a tower you live under any more. It is a workshop you can walk into.

The fire is lit. Forge something.

## What you now know

- The whole arc, end to end: dtypes → storage → tensors → typed axes → ops → SIMD kernels → autograd → training (with and without gradients) → real-time audio → model files and quantization → transformers → inference tricks → low-bit → CPU training → the discipline that holds it together.
- Six graded ways to make the knowledge permanent, each with a documented path in the repo — from running the examples to sending a PR that survives both regression tracks.
- Where an op *lives*: spec in `src/backend/ops.zig`, vector body in `src/backend/vector/primitives.zig`, eager entry in `src/exec/`, derivative in `src/ag/backward.zig`, facade in `src/ag/tensor.zig` — and the oracles that check each station.
- The reading list: the repo's own docs, and the acknowledged projects that built the road first.
- What the name means, and why it fits.

## Explore the source

- `docs/RUNNING-MODELS.md` — start project 1 tonight.
- `src/backend/ops.zig` — the semantic spec of the pointwise library; the cleanest place to begin project 2.
- `src/backend/quant/cold_tests.zig` — what "bit-exact against embedded goldens" looks like; the bar for project 3.
- `docs/PORTING.md` — the method, before you port anything.
- `examples/nam/README.md` — the capture-to-profile walkthrough for project 5.
- `CONTRIBUTING.md` — two pages; read them before your first PR, not after.
- `docs/THIRD-PARTY-NOTICES.md` — the full provenance inventory; the model for crediting whatever you build on.

## Exercises

1. **The victory lap.** Run the core gate set on your machine — `zig build test`, `zig build test -Dbackend=scalar`, `zig build arch-check`, `zig build snippet-check` — and time each ([Chapter 16](16-the-craft.md) lists the complete CI matrix). You are reproducing, on your own hardware, the evidence every chapter of this course leaned on.
2. **Write the report.** Pick any change you made during the course exercises — even a one-line one — and write the PR report `CONTRIBUTING.md` asks for: exact commands, machine and backend configuration, model and quant, failures and skips. Compare it with what you would have written before Chapter 16.
3. **Read one design record end to end** — `docs/MEMORY-MODEL.md` or `docs/SPECULATIVE.md` — and summarize, in one paragraph, the alternative it rejected and the evidence it recorded for the rejection. Writing decisions down this way is a craft skill in itself, and these two documents are the house style.
4. **Pick your project.** Choose one of the six in §17.2, scope its first verifiable milestone — an oracle, not a feature — and build that milestone first. You know why.

---

[Previous: The craft — building a library that can be trusted](16-the-craft.md)
