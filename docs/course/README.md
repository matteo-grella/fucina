# Forging Deep Learning in Zig

*A book-length course that rebuilds the journey of
[Fucina](../../README.md) — Italian for **forge**, a CPU-first
tensor/autograd runtime and LLM inference engine written in pure Zig 0.16 —
from a dtype enum to a live guitar amp and a chatting transformer.*

## What this course is

Open any modern deep-learning project and count the languages between you
and the silicon: Python on top, a framework below it, C++ under that, CUDA
or a vendor BLAS at the bottom. This course takes the other path. Over
eighteen chapters it builds — and reads, line by line — a complete
deep-learning stack in **one language, top to bottom**: tensors, typed
axes, an eager operation library, SIMD kernels, autograd, optimizers,
evolution strategies, a real-time neural guitar amplifier, GGUF and
quantization, a transformer, inference tricks, the ternary frontier, and
LoRA fine-tuning of real language models — all on the CPU, all in Zig.

Fucina is eager: every op executes the moment the model code calls it, on
real buffers. There is no graph to build, plan, or compile first — *what
you read is what runs*. That property is what makes a course like this
possible at all: every claim in the text points at a function you can jump
to in your editor and single-step in a debugger.

The course is not documentation for the library (that is
[`docs/REFERENCE.md`](../REFERENCE.md)). It is the reconstruction of how a
library like this comes to exist — each design decision arriving as the
answer to a problem you have just watched appear — with clean code,
verification discipline, and CPU-first performance as the constant threads.

## Who it is for, and how to read it

Two kinds of readers are the course's students, and a third is its
toughest critic — the text is written for all three:

- **You know ML (Python, PyTorch, papers) but not Zig.** Read in order.
  [Chapter 1](01-just-enough-zig.md) is written for you;
  [Chapter 2](02-just-enough-ml.md) you can skim. Throughout the book,
  `> **Zig note**` asides catch the language details a Pythonista would
  trip on. Your recurring reward: things your framework does invisibly —
  memory, dispatch, parallelism — become things you can see and point at.
- **You know systems programming (C, C++, Rust, Zig) but not ML.** Skim
  [Chapter 1](01-just-enough-zig.md) (but read its comptime sections), then
  take [Chapter 2](02-just-enough-ml.md) slowly — it is your on-ramp.
  `> **ML note**` asides carry the concept definitions. Your recurring
  reward: ML stripped of framework mystique turns out to be numerical code
  you already know how to reason about.
- **You know both.** Move fast and out of order — the
  [fast tracks](#fast-tracks) below and §0.9 of the
  [introduction](00-introduction.md) suggest entry points. The asides are
  not for you; skip them without guilt.

**Honest expectations.** The early chapters are gentle; the later ones are
not, and the text says so when it happens.
[Chapter 12](12-a-transformer-from-scratch.md) calls itself the hardest
chapter so far and means it; Part V in general assumes you absorbed Parts
II and III rather than skimmed them. Nothing is hand-waved — which is
precisely why some sections reward a second read. Every chapter ends with
**What you now know** (a checkpoint), **Explore the source** (files worth
opening), and **Exercises** (two to five, easy to hard).

## Prerequisites

- **A computer with Zig 0.16.0.** Exactly that version — `zig version` must
  print `0.16.0`; other versions do not build. It is a single archive from
  [ziglang.org/download](https://ziglang.org/download/): unpack it, put
  `zig` on your `PATH`, done.
- **No GPU.** CPU-first is the point, not a workaround. Everything in this
  course — including training — runs on an ordinary laptop. (A GPU offload
  seam exists; it gets a section, not a chapter.)
- **No Python, no C++ toolchain, no framework.** There is nothing
  underneath the library but the Zig standard library and your CPU.

The smoke test, from the repo README:

```sh
git clone https://github.com/matteo-grella/fucina
cd fucina
zig build test          # unit tests, no model files needed
```

When that passes, you hold a working forge. The unit tests and the early
chapters need no model files; the language-model chapters (Part V) run real
GGUF checkpoints that you download separately when you get there. One habit
to adopt immediately: build with `-Doptimize=ReleaseFast` whenever speed
matters — Debug builds are 10–50× slower.

## The course, part by part

### Part I — Foundations

| # | Chapter | What it covers |
|---|---------|----------------|
| 00 | [Introduction: why deep learning in Zig?](00-introduction.md) | The tower problem, one readable language top to bottom, Zig's audio origins, CPU-first as a philosophy, the spaGO lineage, honest expectations, and the map of the course. |
| 01 | [Just enough Zig](01-just-enough-zig.md) | Exactly the Zig needed to read and rebuild a tensor library — allocators, error unions, `defer`, optionals, comptime, `@Vector`, `test` blocks, the build system — unfolded from one real twenty-seven-line program. |
| 02 | [Just enough machine learning](02-just-enough-ml.md) | ML from first principles: a model is a function with knobs, loss measures wrongness, gradient descent turns them — plus tensors, layers, softmax and cross-entropy, the chain rule worked by hand, and the two-spirals dataset that recurs all course long. |

### Part II — The tensor core

| # | Chapter | What it covers |
|---|---------|----------------|
| 03 | [Tensors from scratch](03-tensors-from-scratch.md) | Build a miniature tensor from nothing — dtype tag, flat refcounted buffer, shape and strides, zero-copy views, a buffer pool — then meet the real `src/dtype.zig`, `src/storage.zig`, `src/tensor.zig` and the recorded memory-model rationale. |
| 04 | [Axes with names: types that know their shape](04-axes-with-names.md) | The library's signature idea: axis names carried in the type — `Tensor(.{ .batch, .in })` is a different type from `Tensor(.{ .in, .batch })`, contraction happens by name, misalignment is a compile error — and the course's real introduction to comptime. |
| 05 | [The operation library](05-the-operation-library.md) | The verbs, as Fucina implements them: `ExecContext`, the common op contract, ownership in practice, pointwise ops and broadcasting, reductions, `dot` and einsum, the transformer's verbs, losses, determinism as a design stance, and elemental ops for extension. |
| 06 | [Going fast on CPUs](06-going-fast-on-cpus.md) | The close-to-metal chapter: two backends chosen at compile time, `@Vector` SIMD compiling one source tree to NEON and AVX2, GEMM from three naive loops to a blocked packed kernel, the worker team, and honest numbers with the scalar backend as judge. |

### Part III — Learning

| # | Chapter | What it covers |
|---|---------|----------------|
| 07 | [Autograd: the graph hidden in the values](07-autograd.md) | Automatic differentiation with no tape and no graph object — the live tensors *are* the graph — built twice: a tiny scalar autograd you write yourself, then the real engine with atomic dependency counters, checkpointing, custom VJPs, and finite-difference gradcheck. |
| 08 | [Training: making the machine learn](08-training.md) | Gradients into learning: the anatomy of a training step, cross-entropy's knobs, SGD → momentum → AdamW → Muon and APOLLO, schedules, clipping, accumulation, bit-exact checkpoint resume, mixed precision — and spirals trained end to end as the payoff. |
| 09 | [Training without gradients: evolution strategies](09-training-without-gradients.md) | Learning from rewards alone with forward passes only: seed-regenerated noise (O(1) memory beyond the parameters), `fucina.es.Trainer`, antithetic pairs and reward shaping, the `es_spirals` walkthrough, and a clean-room parity story worth stealing. |

### Part IV — Sound

| # | Chapter | What it covers |
|---|---------|----------------|
| 10 | [The guitar amp: real-time neural audio](10-the-guitar-amp.md) | The flagship: a WaveNet guitar amp running live under a ~1.3 ms per-block deadline — `.nam` profiles, streaming convolution, an audio callback guaranteed never to allocate, the latency budget in real numbers, and training your own profile that interchanges with upstream tooling byte for byte. |

### Part V — Language models

| # | Chapter | What it covers |
|---|---------|----------------|
| 11 | [Model files and quantization](11-model-files-and-quantization.md) | A model is tensors + metadata: GGUF and safetensors, mmap as zero-copy loading, byte-identical writers, why CPU decode is a bandwidth problem, block quantization from Q8_0 up through K-quants, and the export tool that quantizes models bigger than RAM. |
| 12 | [A transformer from scratch](12-a-transformer-from-scratch.md) | The chapter people came for, and honestly the hardest so far: tokenization, the transformer block with RoPE and grouped-query attention, the KV cache, prefill vs decode, sampling, chat templates, and mixture-of-experts — grounded line by line in Qwen3-0.6B. |
| 13 | [Inference tricks](13-inference-tricks.md) | Machinery around the decode loop that makes it faster without changing what the model says: draft-free speculative decoding, batch-N multi-stream decode, constrained decoding, KV reuse across requests, MoE expert streaming for models bigger than RAM, and `lmserve`. |
| 14 | [The low-bit frontier](14-the-low-bit-frontier.md) | Weights that take only three values: the TQ2_0 kernel with no multiplications, what 2.06 bits buys (measured), training ternary via straight-through estimators and ES, PTQTP conversion of an existing model, and how to read a research frontier honestly. |
| 15 | [Training LLMs on your CPU](15-training-llms-on-cpu.md) | The loop closes: LoRA from first principles over a frozen quantized Qwen3 GGUF, the SFT data pipeline, a thirty-step fine-tune whose merged export answers in llama.cpp, ES fine-tuning as the gradient-free twin, and karpathy's nanochat trained from scratch. |

### Part VI — The craft

| # | Chapter | What it covers |
|---|---------|----------------|
| 16 | [The craft: building a library that can be trusted](16-the-craft.md) | Clean code as physics: the enforceable layer stack, the verification religion (parity oracles, the scalar twin, machine-verified doc snippets), honest benchmarking with losses recorded, the porting method, and a checklist that transfers to your own projects. |
| 17 | [Epilogue: your forge](17-epilogue.md) | The journey in one page, projects graded from easy to hard, and a reading list — then the hammer is yours. |

## Fast tracks

The chapters are designed to be read in order, but four shorter paths cover
four common reasons to be here:

- **(a) Learning Zig through real code** —
  [01](01-just-enough-zig.md) (the language, unfolded from a real program) →
  [03](03-tensors-from-scratch.md) (allocators, refcounts, and `errdefer`
  in anger) → [04](04-axes-with-names.md) (comptime carrying an entire
  subsystem) → [06](06-going-fast-on-cpus.md) (`@Vector` and compile-time
  specialization) → [10](10-the-guitar-amp.md) (what "no allocation, no
  locks, no syscalls" buys under a real-time deadline).
- **(b) Classic ML foundations** — [02](02-just-enough-ml.md) (models,
  losses, gradients, by hand) → [05](05-the-operation-library.md) (the DL
  vocabulary as ops) → [07](07-autograd.md) (backpropagation, built twice)
  → [08](08-training.md) (optimizers and the training loop) →
  [09](09-training-without-gradients.md) (learning without gradients at
  all).
- **(c) LLMs** — [11](11-model-files-and-quantization.md) (model files and
  quantization) → [12](12-a-transformer-from-scratch.md) (the transformer
  itself) → [13](13-inference-tricks.md) (making decode fast) →
  [14](14-the-low-bit-frontier.md) (the ternary frontier) →
  [15](15-training-llms-on-cpu.md) (fine-tuning on CPU). Chapters
  [05](05-the-operation-library.md)–[08](08-training.md) are the assumed
  background when the going gets steep.
- **(d) The real-time audio story** — [00](00-introduction.md) §0.3 (Zig
  was born from an audio project) → [10](10-the-guitar-amp.md) (the guitar
  amp, end to end). To train your own amp profile rather than just play
  one, add [07](07-autograd.md) and [08](08-training.md) first.

## How code and numbers work in this course

Every chapter follows the same conventions, so you always know what you are
looking at:

- **Repo code is cited, never invented.** Snippets copied from the
  repository carry their path (e.g. *from `src/tags.zig`*), and often line
  numbers. Snippets quoted from [`docs/REFERENCE.md`](../REFERENCE.md) are
  **machine-verified**: the repo's `zig build snippet-check` gate compiles
  and runs them against the real modules in CI
  ([Chapter 16](16-the-craft.md) §16.4 shows how).
- **Course code is labelled as course code.** The build-it-yourself
  miniatures — a toy tensor, a sixty-line ES trainer, a scalar autograd —
  are fresh teaching code, clearly marked as such, and compile-checked with
  the pinned Zig 0.16.0 whenever they are complete programs. They are
  deliberately minimal; the real thing always follows, cited from the tree.
- **Numbers come from the repo's own documents, named and dated.**
  Benchmarks are quoted from files like [`docs/BENCHMARK.md`](../BENCHMARK.md)
  with their caveats attached — measured, not asserted; snapshots, not
  promises; losses recorded next to wins.
- **Asides serve the two newcomer audiences.** `> **Zig note**` explains
  language details for ML readers; `> **ML note**` explains concepts for
  systems readers; the third audience skips both without guilt. They are
  frequent early and sparse late.
- **Cross-links are relative.** Chapters link to each other as sibling
  files, so the course reads the same on GitHub and in a local clone.
- **Citations are pinned to a moment.** This edition of the course was
  reconciled against the tree at commit `58383e3` (2026-07-17): every
  `path:line` reference was verified there. The library moves; line
  numbers drift first, then signatures. If the text and the source ever
  disagree, **the source wins** — diff against that commit to see what
  changed.

Start with [Chapter 00 — Introduction: why deep learning in
Zig?](00-introduction.md)
