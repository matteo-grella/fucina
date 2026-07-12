# Chapter 00 — Introduction: why deep learning in Zig?

*Part I — Foundations*

Open any modern deep-learning project and count the languages between you and
the silicon. You write Python. The Python calls a framework. The framework
dispatches into C++. The C++ launches CUDA kernels, or calls a vendor BLAS, or
hands a graph to a compiler that rewrites it into something you will never
read. Each layer exists for a good reason, and each layer is a place where
your understanding stops. When the model is wrong, or slow, or runs out of
memory, you are debugging a tower — and you only own the top floor.

This course takes the other path. We are going to build — and read, line by
line — a deep-learning stack written in **one language**, top to bottom: from
the bytes of a tensor buffer to a transformer answering questions, from a
gradient computed by the chain rule to a neural guitar amplifier running live
under a real-time deadline. The language is Zig. The library is
[Fucina](../../README.md) — Italian for *forge* — a CPU-first tensor/autograd
runtime and LLM inference engine whose entire source you can hold in your
head, because there is nothing underneath it but the standard library and the
CPU.

Two kinds of readers are welcome here, and the course is written for both at
once: programmers who have never written Zig (you will learn the language
through a genuinely demanding, genuinely rewarding use case), and programmers
who have never done machine learning (you will build ML from first
principles, in a language that hides nothing). Neither track requires the
other's background. Both require patience: the later chapters are honestly
hard, and we will say so when they are.

There is also a third reader, and it would be false modesty not to address
you directly: you already know systems programming *and* deep learning, and
you are here because a one-language, no-compiler, CPU-first stack sounds
either interesting or wrong. You are not the student this course was framed
for, but you may be the reader it serves best — what it offers you is a
complete, verified, deliberately transparent implementation of everything
your production stack does behind a compiler pass, laid out one seam at a
time where you can argue with it. §0.7 states plainly what this project
does and does not claim; start there, then go wherever your skepticism
points.

## 0.1 The tower problem

The mainstream deep-learning stack is a marvel, and this course is not here
to sneer at it. PyTorch and its siblings serve millions of users across
thousands of hardware configurations; the tower is the price of that
generality. But the tower has costs that matter enormously when your goal is
*understanding*:

- **You cannot read it end to end.** The distance from `model(x)` in Python
  to a fused kernel is several codebases, several build systems, and at least
  two languages away. "What actually happens when I call this?" has no
  short answer.
- **The graph is not your program.** Most stacks either trace your code into
  a graph, compile it, or schedule it through a central engine. The code you
  wrote is an *input* to the thing that runs, not the thing that runs.
- **The performance model is opaque.** Why is this shape fast and that one
  slow? The answer lives in a dispatcher you did not write, choosing among
  kernels you cannot see, on heuristics that change between releases.

Fucina's founding stance is the opposite, stated in the first paragraph of
its README (README.md:3-9):

> Computation is **eager**: every op executes the moment the model code calls
> it, on real buffers. There is no graph to build, plan, or compile first, so
> what you read is what runs, in inference and in training alike.

*What you read is what runs.* That sentence is the course's thesis. Every op
is an ordinary function call. Every intermediate tensor is a value you own
and free. Every kernel is a Zig function you can jump to in your editor. And
— the README continues, pre-empting the obvious objection — "Eager does not
mean slow: on most measured CPU shapes Fucina matches or beats llama.cpp"
(README.md:7-9; the dated, caveated record is `docs/BENCHMARK.md`, and we
will read it honestly in §0.4).

> **ML note** — "Eager" vs "graph" execution is a real axis in framework
> design. Graph systems (TensorFlow 1.x classically; compiler stacks like XLA
> today) first build a description of the whole computation, optimize it,
> then run it. Eager systems execute each operation immediately, like any
> other code. Eager is easier to debug and reason about; graphs enable global
> optimization. Fucina is deliberately, permanently eager — [Chapter 5](05-the-operation-library.md)
> shows what that buys and what it costs.

## 0.2 One language, top to bottom

Here is what the stack looks like when there is no tower. This is the
front-page example from the repository's README (README.md:24-54) — a
two-layer neural network, its forward pass, and the training loop, condensed.
Do not worry about understanding every token yet; that is what the next
fifteen chapters are for. Read it for its *shape*:

```zig
const Model = struct {
    w1: Tensor(.{ .h1, .in }),
    b1: Tensor(.{.h1}),
    w2: Tensor(.{ .class, .h1 }),
    b2: Tensor(.{.class}),
};

fn forward(ctx: *ExecContext, m: *const Model, x: *const Tensor(.{ .batch, .in })) !Tensor(.{ .batch, .class }) {
    var z1 = try x.dot(ctx, &m.w1, .in); // contract .in -> .{ .batch, .h1 }
    defer z1.deinit();
    var s1 = try z1.add(ctx, &m.b1);
    defer s1.deinit();
    var a1 = try s1.tanh(ctx);
    defer a1.deinit();
    var z2 = try a1.dot(ctx, &m.w2, .h1); // contract .h1 -> .{ .batch, .class }
    defer z2.deinit();
    return try z2.add(ctx, &m.b2);
}

// Inference: the defers free each intermediate as soon as it is consumed.
// Training: open an exec scope and the SAME forward trains as-is — the
// scope adopts every intermediate (value + autograd node), each deinit
// becomes a no-op borrow-release, and the step's whole graph stays alive
// until backward(), then is released at once when the scope closes.
const scope = ctx.openExecScope();
defer ctx.closeExecScope(scope);
const logits = try forward(ctx, &model, &x);
const loss = try logits.crossEntropy(ctx, .class, labels);
try loss.backward(ctx);
try opt.step(ctx);
```

Four things to notice, each of which becomes a chapter:

1. **The axes have names, and the names live in the types.**
   `Tensor(.{ .batch, .in })` is a *different type* from
   `Tensor(.{ .in, .batch })`. Contraction is by axis name (`.dot(ctx, &m.w1,
   .in)` contracts the `.in` axis), the result's tag set is computed at
   compile time, and a misaligned contraction is a compile error — "not a
   runtime shape crash three layers deep" (README.md:16-20). The README
   offers the right historical rhyme (README.md:56-60): "The shape discipline
   lives in the program text, the way it did in Fortran — `real A(n,m)` told
   you the rank before you read a single loop." [Chapter 4](04-axes-with-names.md)
   builds this machinery.
2. **Memory is explicit and visible.** Every intermediate is freed by a
   `defer z1.deinit()` — you can see exactly when each buffer dies. There is
   no garbage collector and no hidden caching layer; there is a buffer pool
   whose design (and the arena design it beat) is documented with
   measurements in `docs/MEMORY-MODEL.md`. [Chapter 3](03-tensors-from-scratch.md)
   starts here.
3. **The same forward function infers and trains.** No `model.train()` mode,
   no separate graph for backward. Open an *exec scope* and the identical
   code records what it needs; call `backward()` and gradients flow. How that
   works — and why it needs no graph object at all — is
   [Chapter 7](07-autograd.md).
4. **It is all just Zig.** `try`, `defer`, structs, functions. The error
   handling is the language's error handling. If you can read this file, you
   can read the library.

> **Zig note** — If `defer`, `try`, and `.{ .h1, .in }` are new to you: they
> are, respectively, a statement that runs at scope exit (deterministic
> cleanup without destructors), an early-return on error (errors are values
> in Zig, not exceptions), and an anonymous literal (here, a tuple of enum
> literals used as axis tags). [Chapter 1](01-just-enough-zig.md) teaches
> exactly the Zig this library needs — no more.

## 0.3 Zig was born from audio — and this course ends with a guitar amp

The language choice is not arbitrary, and its own origin story fits this
course almost too well. Andrew Kelley conceived Zig while building
[Genesis](https://github.com/andrewrk/genesis), a digital audio workstation,
after C and C++ — and their library ecosystems — fell short for low-latency,
reliable audio software; before that he had built the
[libgroove](https://github.com/andrewrk/libgroove) audio library in C. He
announced Zig on February 8, 2016
([andrewkelley.me/post/intro-to-zig.html](https://andrewkelley.me/post/intro-to-zig.html)).
Zig, in other words, exists because someone needed to process sound in real
time and wanted a language that was fast, explicit about memory, and free of
hidden control flow.

This course closes that circle deliberately. Its flagship application —
[Chapter 10](10-the-guitar-amp.md) — is a **real-time neural guitar
amplifier**: a WaveNet loaded from a `.nam` profile, processing a live
guitar signal inside an audio callback, in Zig, on a laptop. The audio
callback is the harshest environment in userspace programming: you have a
fixed budget of microseconds per buffer, and an allocation, a lock, or a
garbage-collection pause is an audible glitch. It is the perfect stress test
for the claim that a deep-learning stack can be an ordinary, predictable
program — and the strongest argument in this book for what "close to the
metal" buys.

Between here and there, every property that makes the amp possible gets
built and explained: explicit allocators ([Chapter 1](01-just-enough-zig.md)
and [3](03-tensors-from-scratch.md)), allocation-free kernels
([Chapter 6](06-going-fast-on-cpus.md)), a trained model
([Chapter 8](08-training.md)), and streaming convolutions
([Chapter 10](10-the-guitar-amp.md) itself).

## 0.4 CPU-first is a philosophy, not a limitation

Fucina runs on CPUs. Not "CPU as a fallback while the real code targets
GPUs" — CPU-first, as a design center. The reasons are worth stating,
because they are also the reasons this course can exist:

- **It is the machine you already own.** Everything in this course runs on a
  laptop. No cloud account, no driver stack, no CUDA version matrix. The
  models in the repo's table — from a 0.6B chat model to out-of-core
  mixture-of-experts giants — run on machines you can own: the giants on a
  single 64 GB box (README.md:92-95), not a datacenter.
- **Latency and locality.** For interactive work — a chat turn, a live audio
  buffer — the data is already in your RAM. The guitar amp is the extreme
  case: sound cannot round-trip to a datacenter.
- **Determinism and reproducibility.** One machine, one thread pool, a
  repo-owned counter-based RNG: the same seed gives the same run. Fucina
  treats determinism as a contract: "dropout masks, APOLLO projections, ES
  noise are regenerated from seeds, not serialized"
  (docs/DEVELOPMENT.md §1.9) — which is also what makes its testing
  discipline possible.
- **The performance ceiling is higher than the folklore says.** This is the
  claim that needs numbers, so here is exactly how the project states them.

`docs/BENCHMARK.md` opens with two ground rules (docs/BENCHMARK.md:13-19):
every number carries its hardware and measurement conditions, and "**Losses
are recorded as plainly as wins.**" The whole record "is one snapshot, taken
as of 2026-07-04" (docs/BENCHMARK.md:9) against a pinned llama.cpp build,
same GGUF file, same thread count, CPU-only on both sides. Within those
caveats, the README's summary (README.md:155-164): on an Apple M1 Max, of
236 paired sweep cells across several model families, Fucina was faster in
221 and at parity in 13 — dense prefill geomeans 1.18–1.81x per format —
with two cells on llama.cpp's side, one of them Qwen3.5-0.8B at prompt
length 32 (0.86x, honestly kept on the scoreboard). On an x86 Raptor Lake
box, Fucina won all dense quantized formats (medians 1.32–1.95x) while
"llama.cpp decisively wins MoE small-batch prefill."

Treat every one of those numbers as what the repo says it is: a dated,
machine-specific snapshot, reproducible via the paired benchmark gate
(`tools/bench_gate.py`), and aging from the moment it was taken —
"Benchmarks age. llama.cpp moves fast; the dated records in
`docs/BENCHMARK.md` are snapshots, not eternal claims" (README.md:240-241).
[Chapter 6](06-going-fast-on-cpus.md) shows how the speed is achieved;
[Chapter 16](16-the-craft.md) shows how it is *measured* without
self-deception — the protocol is at least as interesting as the numbers.

Two structural facts to fix in your mind now, because loose talk about them
causes real confusion later:

- Fucina has exactly **two CPU backends**: `scalar`, the slow, obvious
  reference implementation that serves as the executable specification, and
  `native`, the fast SIMD one that must agree with it. (You may see `cpu` in
  older invocations; it is a deprecated alias for `scalar`.) Optional CBLAS
  providers supply GEMM as a *provider* choice within a backend, and the
  Metal/CUDA GPU paths are an *offload seam* for specific shapes — neither
  is a third backend, and the README is explicit that the Metal offload "is
  not a general GPU runtime" (README.md:227-233).
- The scalar/native pair is not an implementation detail; it is the
  verification strategy. Every fast kernel is held to the boring one by a
  parity suite. That idea — *the spec is a program* — recurs through the
  whole course.

> **ML note** — "GEMM" is GEneral Matrix Multiply, the workhorse of deep
> learning: most of a neural network's compute is matrix multiplication, so
> most of the performance story is the GEMM story.
> "Prefill" and "decode" are the two phases of LLM inference — processing
> the prompt in bulk vs generating one token at a time — and they have very
> different performance characters. Both get proper treatment in
> [Chapter 12](12-a-transformer-from-scratch.md).

## 0.5 Origins: the graph implicit in the values

Fucina did not begin in Zig. Its central idea has a lineage, told in the
README's Origins section, which is worth quoting whole (README.md:245-260):

> Fucina grew out of autograd concepts I first explored in Go with
> [spaGO](https://github.com/nlpodyssey/spago) — above all the idea that the
> graph should be **implicit in the values themselves**: no graph object, no
> tape, no persistent engine. Each result carries a pointer to the operation
> that produced it, and `backward()` discovers the topology by walking those
> pointers. spaGO executed that idea the Go way: one goroutine per node,
> each blocking until its gradient contributions arrived, the runtime
> scheduler absorbing the wait. Zig has no goroutines, so Fucina keeps the
> idea and rethinks the execution: every node carries an atomic dependency
> counter, and its gradient fires only when the counter drains — concurrent,
> on a bounded worker pool, no blocked workers (`src/ag/`). (AFAIK) Mainstream
> stacks route backward through a central engine over an explicit node graph
> or a trace; here the live tensors *are* the graph.
>
> As for the language: I wanted to stay as close to the metal as possible, and — honestly — I was
> also looking for a good excuse to finally learn Zig. This project is it.

Note the "(AFAIK)" — the hedge is the author's, and we keep it: the claim is
about the designs the author knows, not a survey of every engine ever built.
But the idea itself is the most elegant thing in this book, and
[Chapter 7](07-autograd.md) is built around it: there is no graph *data
structure* anywhere. A tensor that came out of an operation remembers which
operation; calling `backward()` on the loss walks those pointers backwards.
The "computation graph" that frameworks reify as an object is, here, just…
the values you already have, and the pointers between them. You will build a
miniature version yourself before meeting the real engine.

The last two lines of the quote also set this course's tone. The library
exists partly because its author wanted to *learn the language by building
something real*. That is precisely the deal this course offers you.

## 0.6 A library grown through real applications

Fucina's `examples/` directory is not a demo folder. The README states the
growth model plainly (README.md:103-110):

> These applications live in `examples/` and each will eventually graduate
> into its own repository. Meanwhile they are here for convenience, and not by
> accident: with the Tensor core in place, Fucina grows and gets tested through real
> applications, so the runtime and the things built on it develop side by
> side.

What runs today (condensed from the README's table, README.md:74-95): chat
LLMs dense and mixture-of-experts (Qwen3, DeepSeek V2/V3 and V4 Flash,
GLM-4.5, Gemma 4, Qwen3.5), a text-diffusion decoder, karpathy's nanochat
ported whole and *trained from scratch on CPU*, speech-to-text (Parakeet),
text-to-speech with voice cloning (OmniVoice), an open-vocabulary detection
VLM, face detection and recognition — plus, from the quick start rather
than the table, an OpenAI-compatible HTTP server (README.md:128-130) — and
the Neural Amp Modeler. MoE models bigger than RAM are first-class: expert
streaming is "how the 142 GB Qwen3-235B and the 164.6 GB V4 Flash decode on
a 64 GB machine" (README.md:92-95).

Each of those applications forced something into the core — new ops, new
dtypes, new scheduler behaviour — and each earned its place the same way
(README.md:97-100):

> Every family is validated against its reference implementation, and that
> discipline is the core of the project: token-ID-exact tokenizers vs
> `llama-tokenize`, logit-parity oracles vs llama.cpp, byte-exact quantization
> encoders vs ggml, byte-identical GGUF re-emit.

This is the second thread that runs through the whole course, alongside
"what you read is what runs": **you don't optimize what you can't verify**
(that phrasing is from `docs/PORTING.md`). Fucina never claims a model works
because the output "looks right"; it claims parity against a pinned
reference, checked by a gate that exits nonzero on failure. Scalar-vs-native
kernel parity, finite-difference gradient checks, golden-parity optimizer
tests, machine-verified documentation snippets — the verification apparatus
is a character in this story, introduced piece by piece and assembled in
[Chapter 16](16-the-craft.md).

> **Zig note** — Even the documentation is under test: `zig build
> snippet-check` extracts the runnable code blocks from `docs/REFERENCE.md`
> and compiles and runs them against the real modules. When this course
> quotes a REFERENCE.md snippet, you are reading machine-verified code.

## 0.7 Honest expectations

This course teaches from a real, young codebase, and inherits its honesty
obligations. Before you invest a book's worth of attention, know exactly
what Fucina is and is not:

- **It is not a production-ready product.** The architecture document grades
  itself: "**production-oriented core, not production-ready product**"
  (docs/ARCHITECTURE.md:712-713). The core is coherent and machine-enforced;
  the productization gaps (API contract, session lifecycle, platform
  coverage) are listed right above that sentence.
- **The API is not stable.** "This is a young codebase published in the
  open, not a 1.0 library. Expect churn" (README.md:234-235). There is no
  package manifest and no semantic versioning. Every signature in this
  course is *today's code*, cited by path so you can diff it against
  tomorrow's.
- **Two ISAs are tuned.** Apple Silicon (NEON/dotprod) and modern x86-64
  (AVX2/AVX-VNNI); the scalar backend covers everything else, correctly but
  slowly (README.md:227-229).
- **Model weights are not included**, and each family carries its own
  license (README.md:236-239).
- **Benchmarks age** (README.md:240-241) — every number you meet in this
  course is a dated snapshot with a machine attached, never a universal
  claim.

Three more things belong in this section, because the experienced reader
will ask about all of them, and because they are true.

**What this course is not claiming.** Fucina is "deliberately **eager and
local**: no global graph object, no fusion pass, no compiler layer"
(README.md:64-66). On the other side of that decision sits an entire world
of deep-learning-systems research and engineering — graph capture, operator
fusion, memory planning, autotuned code generation; the machinery inside
XLA, TVM, and `torch.compile`, and the accumulated best practices of teams
who have spent a decade on exactly those layers. This project does not
engage that world, and it does not claim to be state of the art against it.
That is a scope decision, made deliberately and with full awareness of what
lives on the other side — not a judgment of that work, and not an accident
of ignorance. What Fucina claims instead is narrower and measured: that on
the CPU shapes it targets, the eager-and-local design is competitive with a
mature reference implementation (chapter 6 shows the protocol, and the
losses alongside the wins), while remaining a codebase a single person can
hold in their head. That second property is not a consolation prize; here,
it is the product — and for the reader who already knows what a fusion pass
buys, a coherent stack with *zero* of them is a useful object: it shows
what those layers cost in legibility, and it makes visible exactly what
they would have to earn back.

**Whose work this is.** Fucina is one person's effort — at the time of
writing, one human, working "with strong assistance from agentic coding
systems, with humans leading the ideas, the testing, and the debugging"
(README.md:262-266). The course inherits both sides of that fact: the
coherence of a codebase where every layer was shaped by one set of hands
and one set of convictions, and the limits of what one person can build,
tune, and verify. The verification discipline you will meet throughout —
parity oracles, golden tests, machine-checked documentation — exists in
large part *because* of those limits: it is how a small effort earns trust
it cannot buy with headcount.

**How this course was made.** The same candor applies to the book in your
hands. This course was generated with AI — the same class of agentic
systems that assisted Fucina's development — working retrospectively over
the library as it stands today: reading the source, `docs/REFERENCE.md`,
and the design documents, and reconstructing from them the decisions, and
the reasons behind them, that shaped Fucina so far. "As it stands today"
means exactly that: a significant effort has been made to reach this
point, and any of the decisions you will read about could still be
revised — this is a young library under active development, not a closed
book. Two consequences follow, and you should know both. First, the "from
scratch" journey of Parts II–V is a *teaching order*, not a development
diary: it is how the ideas stack most legibly when reconstructed after
the fact, not a claim that the code was written in that sequence. Second, accuracy was pursued the way the library
itself pursues correctness — mechanically, not by good intentions. Every
claim about the repo is cited by path (usually with line numbers) so you
can check it; code excerpts taken from `docs/REFERENCE.md` are
machine-verified against the real modules in CI; fresh course code
compiles under the pinned Zig 0.16.0; every performance number is quoted
from a repo document together with its machine and date. Where the text
goes beyond the repo — history, analogies, general ML background — it says
so. What to emphasize was an editorial choice: the concepts judged most
useful to this course's audiences, selected from what the codebase
actually contains. If you ever find the text and the source in
disagreement, the source wins — and the citation is there precisely so you
can catch it.

One more expectation, about the language itself: **Zig 0.16 is not the Zig
of most online tutorials.** The toolchain is pinned — "Requires Zig 0.16.0
… other versions will not build" (README.md:114-115) — and 0.16 changed
several standard-library idioms you may find described differently
elsewhere. When in doubt, imitate this repository, not a blog post from two
years ago. [Chapter 1](01-just-enough-zig.md) teaches the current idioms
directly.

And about difficulty: Parts I–III are carefully paced for both audiences.
Part IV (the guitar amp) is a joy with everything before it in hand. Part V
— transformers, quantization, inference tricks, the low-bit frontier — is
genuinely hard, and pretending otherwise would waste your time. The text
will always tell you which sections are load-bearing and which you can skim
on a first pass.

## 0.8 The map of the course

Eighteen chapters in six parts. Each is written to be read in order, but the
cross-links are dense enough to support other paths (§0.9).

**Part I — Foundations**

| Ch | Title | What it gives you |
| --- | --- | --- |
| 00 | this chapter | the thesis, the map, the expectations |
| [01](01-just-enough-zig.md) | Just enough Zig | the language, feature by feature, each motivated by the library's needs: allocators, slices, errors, `defer`, comptime, `@Vector`, the build system |
| [02](02-just-enough-ml.md) | Just enough machine learning | what "learning" means: models as parameterized functions, loss, gradients, the chain rule by hand — and the two-spirals dataset, our fruit fly |

**Part II — The tensor core**

| Ch | Title | What it gives you |
| --- | --- | --- |
| [03](03-tensors-from-scratch.md) | Tensors from scratch | dtypes, refcounted storage, shapes and strides, zero-copy views; build a minimal tensor yourself, then meet `src/tensor.zig` |
| [04](04-axes-with-names.md) | Axes with names | the crown jewel: comptime axis tags, shape errors at compile time — and honestly, which errors stay at runtime |
| [05](05-the-operation-library.md) | The operation library | `ExecContext`, the op contract, broadcasting, reductions, `dot` and einsum; the deep-learning vocabulary as a library of eager ops |
| [06](06-going-fast-on-cpus.md) | Going fast on CPUs | `@Vector` SIMD, the blocked GEMM, the thread pool, scalar-as-oracle, and the benchmark protocol |

**Part III — Learning**

| Ch | Title | What it gives you |
| --- | --- | --- |
| [07](07-autograd.md) | Autograd | the graph hidden in the values; build a tiny autograd, then read the real one; gradcheck as the calculus referee |
| [08](08-training.md) | Training | losses, SGD → AdamW → Muon/APOLLO, schedules, checkpoints — and the full spirals walkthrough, every line explained |
| [09](09-training-without-gradients.md) | Training without gradients | evolution strategies as a first-class citizen: forward passes only, seed-regenerated noise |

**Part IV — Sound (the flagship)**

| Ch | Title | What it gives you |
| --- | --- | --- |
| [10](10-the-guitar-amp.md) | The guitar amp | WaveNet, streaming convolution, the real-time latency budget in real numbers, and playing guitar through a neural network you understand completely |

**Part V — Language models**

| Ch | Title | What it gives you |
| --- | --- | --- |
| [11](11-model-files-and-quantization.md) | Model files and quantization | GGUF, safetensors, and why low-bit weights are a bandwidth story, not (only) a memory story |
| [12](12-a-transformer-from-scratch.md) | A transformer from scratch | the chapter you came for: tokenizer, attention, RoPE, the KV cache, sampling, MoE — grounded in the Qwen3 source |
| [13](13-inference-tricks.md) | Inference tricks | speculative decoding without a draft model, multi-stream decode, constrained output, KV reuse, expert streaming, serving |
| [14](14-the-low-bit-frontier.md) | The low-bit frontier | ternary weights, straight-through estimators, PTQTP — the research edge, framed as research |
| [15](15-training-llms-on-cpu.md) | Training LLMs on your CPU | LoRA end-to-end on a quantized GGUF; nanochat trained from scratch; what CPU training is honestly for |

**Part VI — The craft**

| Ch | Title | What it gives you |
| --- | --- | --- |
| [16](16-the-craft.md) | The craft | how a library like this is actually built: enforced layering, the verification religion, honest benchmarking, the porting method |
| [17](17-epilogue.md) | Epilogue: your forge | the journey in one page, and graded project ideas for what you build next |

## 0.9 How to read this book

**If you come from ML (Python, PyTorch, papers) and Zig is new:** read in
order. Chapter 1 is written for you; Chapter 2 you can skim, pausing only on
the notation. Throughout the book, `> **Zig note**` asides catch the
language details a Pythonista would trip on. Your recurring reward: things
your framework does invisibly — memory, dispatch, parallelism — become
things you can see and point at.

**If you come from systems programming (C, C++, Rust, or Zig itself) and ML
is new:** skim Chapter 1 (but do read its comptime section — Fucina's use of
comptime is beyond most codebases), then take Chapter 2 slowly; it is your
on-ramp, and everything later builds on it. `> **ML note**` asides carry the
concept definitions. Your recurring reward: ML stripped of framework
mystique turns out to be numerical code you already know how to reason
about, plus a handful of genuinely deep ideas (the chain rule, attention)
that arrive with their motivations attached.

**If you are fluent in both** — systems code shipped, models trained — you
can move fast and out of order. Read §0.7 above if you skipped it, then go
straight to where your interest (or doubt) points: [Chapter 4](04-axes-with-names.md)
for shape discipline moved into the type system, [Chapter 6](06-going-fast-on-cpus.md)
for what portable SIMD and a blocked GEMM do against a mature baseline
(losses recorded), [Chapter 7](07-autograd.md) for an autograd with no tape
and no engine, [Chapter 13](13-inference-tricks.md) for speculative decoding
without a draft model. The `Zig note` and `ML note` asides are not for you;
skip them without guilt. Read the early chapters last, if at all — as a
statement of method rather than instruction.

**Either way, run things.** The course's code comes in two flavours, always
labelled: verbatim repo code, cited by path (when it comes from
`docs/REFERENCE.md`, it is machine-verified against the real modules); and
minimal build-it-yourself course code, which compiles under the pinned Zig
0.16.0. Get the toolchain now — the quick start is the README's
(README.md:117-131):

```sh
git clone https://github.com/matteo-grella/fucina
cd fucina
zig build test          # unit tests, no model files needed
```

That `zig build test` needs no model weights and no network beyond the
clone. When it passes, you hold a working forge. One habit to adopt
immediately, straight from the README: build with `-Doptimize=ReleaseFast`
whenever speed matters — "Debug is 10–50x slower" (README.md:133-134).

## What you now know

- The course thesis: a deep-learning stack in **one language, top to
  bottom, on your own CPU** — no Python layer, no C++ layer, no graph
  compiler between you and the machine.
- Fucina is *eager*: "what you read is what runs", in inference and
  training alike — and the same forward function serves both, via exec
  scopes (README front example).
- Zig itself was born from a real-time audio project — Andrew Kelley's
  Genesis DAW — and this course deliberately closes that circle with a
  real-time neural guitar amp in Chapter 10.
- The performance claims are real but disciplined: dated, machine-specific,
  paired-run snapshots (2026-07-04, `docs/BENCHMARK.md`), losses recorded
  as plainly as wins — never universal claims.
- Fucina has exactly two CPU backends — `scalar` (the reference oracle) and
  `native` (the fast one); BLAS is a GEMM provider option and Metal/CUDA an
  offload seam, not backends.
- The autograd design descends from spaGO: the graph is implicit in the
  values — each result points at the op that produced it, and `backward()`
  walks the pointers.
- The library grew through real applications, each validated against a
  pinned reference implementation; verification discipline is the core of
  the project, and of this course.
- Honest expectations: a production-oriented core, not a production-ready
  product; unstable API; pinned Zig 0.16.0; hard later chapters.

## Explore the source

- `README.md` — read it whole, now that you have the frame: the front
  example, the model table, the Origins section, the acknowledgments (the
  road others built first).
- `docs/ARCHITECTURE.md` — the layer-by-layer map of the tree; its band
  table is, read bottom-up, the build order this course follows.
- `docs/BENCHMARK.md` — the measurement protocol and the scoreboard; read
  the two ground rules at the top before any number.
- `examples/spirals.zig` — the smallest complete train/checkpoint/infer
  program in the repo, and the running example of Chapters 2 and 8.
- `docs/MEMORY-MODEL.md` — ownership rules and the buffer-pool-vs-arena
  adjudication: a masterclass in documenting a rejected alternative.

## Exercises

1. **(easy)** Install [Zig 0.16.0](https://ziglang.org/download/) — the
   exact version; the toolchain is pinned — clone the repository, and run
   `zig build test`. No model files are needed. Note how long a clean build
   takes: that is the whole stack compiling.
2. **(easy)** Run the two-spirals demo: `zig build spirals
   -Doptimize=ReleaseFast`. Per its header (`examples/spirals.zig:1-14`), it
   trains a small MLP with five different optimizers, proves bit-exact
   training resumption from a checkpoint, and reports inference accuracy.
   You will understand every line by the end of [Chapter 8](08-training.md).
3. **(medium)** Read the Scoreboard section of `docs/BENCHMARK.md` and find
   two entries where llama.cpp wins. For each, note what the record says
   about *why* and under what conditions it was measured. This calibrates
   you for every performance claim in the book.
4. **(medium)** In the forward function of §0.2, work out on paper what the
   type of each intermediate (`z1`, `s1`, `a1`, `z2`) must be, and what
   should happen at compile time if you changed the first contraction to
   `x.dot(ctx, &m.w1, .batch)`. Keep your answer; [Chapter 4](04-axes-with-names.md)
   settles it.
5. **(harder, optional — needs ~1 GB of disk and network)** Follow the
   README quick start to download Qwen3-0.6B and chat with it:
   `zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf
   --chat "What is the capital of France?" --no-think`. Every stage of what
   just happened — GGUF parsing, tokenization, attention, sampling — is a
   chapter in Part V.

---

[Previous: Course index](README.md) · [Next: Chapter 01 — Just enough Zig](01-just-enough-zig.md)
