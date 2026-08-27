# Fucina

[![CI](https://github.com/matteo-grella/fucina/actions/workflows/ci.yml/badge.svg)](https://github.com/matteo-grella/fucina/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-manual-F26B1F)](https://matteo-grella.github.io/fucina/docs/)
[![Zig](https://img.shields.io/badge/Zig-0.16.0-F7A41D)](https://ziglang.org/download/)

**Fucina** (Italian for *forge*) is a CPU-first tensor/autograd library
written in pure **Zig 0.16**. Axes have names checked at compile time, and
computation is **eager**: every op executes the moment your code calls it,
on real buffers — no graph to build, plan, or compile first, so what you
read is what runs, in inference and in training alike. There is no C/C++
build system and no Python runtime dependency: Zig vector kernels, with
CBLAS providers and a GPU offload (Metal or CUDA) as opt-in accelerators
for matrix multiplication.

To prove the library on real workloads, this repository also ships complete
**applications built on it** (see
[what the library enables today](#what-the-library-enables-today)):
LLM chat and serving (Qwen3, DeepSeek, GLM,
Gemma, ...), speech, vision, and audio models — each validated against its
reference implementation and benchmarked against llama.cpp, which it
matches or beats on most measured CPU shapes
([docs/BENCHMARK.md](docs/BENCHMARK.md), losses included). The
applications use the library; they are not the library, and they will
graduate to their own repositories.

**[The manual](https://matteo-grella.github.io/fucina/docs/)** is the
rendered documentation: the machine-verified API reference, the guides, and
[*Forging Deep Learning in Zig*](https://matteo-grella.github.io/fucina/docs/course/),
a book-length course that rebuilds this library from a dtype enum to a live
guitar amp and chatting language models.

## What it looks like

Axis tags and rank are **comptime facts**: `Tensor(.{ .batch, .in })` is a
different type from `Tensor(.{ .in, .batch })`, contraction is by axis
*name*, the result's tag set is computed at compile time, and a misaligned
contraction is a compile error — not a runtime shape crash three layers
deep. The pattern, condensed (see `examples/spirals/main.zig` and the production
trainer in `src/models/qwen3/train.zig`):

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
var loss = try logits.crossEntropy(ctx, .class, labels, .{});
try loss.backward(ctx);
try opt.step(ctx);
```

The shape discipline lives in the program text, the way it did in Fortran —
`real A(n,m)` told you the rank before you read a single loop — and Zig's
comptime makes it free: the tags exist only in the type system and compile
away entirely.

## Getting started

Requires [Zig 0.16.0](https://ziglang.org/download/) — the toolchain is
pinned (`build.zig.zon` enforces the minimum); other versions will not
build. Fucina is an ordinary Zig package:

```sh
zig fetch --save git+https://github.com/matteo-grella/fucina#v0.3.0
```

```zig
// build.zig
const fucina_dep = b.dependency("fucina", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("fucina", fucina_dep.module("fucina"));
exe.root_module.addImport("fucina_models", fucina_dep.module("fucina_models")); // LLM stack; optional
exe.root_module.addImport("fucina_serving", fucina_dep.module("fucina_serving")); // HTTP serving transport; optional
```

The fetched package is the library surface only (`src/`, the vendored
llguidance source, build files — about 11 MB): everything a dependency
build needs and nothing else. The model runners, example applications,
docs, and tools are repo development surface — to run those, clone the
repo instead (next section).

If you do want the complete repository as your dependency (say, to hack
on fucina and your project together), skip `zig fetch`: clone the repo
(or add it as a git submodule) next to your project and point
`build.zig.zon` at the directory —

```zon
.fucina = .{ .path = "../fucina" },
```

A path dependency uses the directory exactly as it is on disk, so
everything a clone has is there.

A first program, verbatim from
[the reference's §1.4](https://matteo-grella.github.io/fucina/docs/reference/01-introduction-and-mental-model/#14-a-first-program),
where CI compiles and runs this exact block on every push:

```zig
const std = @import("std");
const fucina = @import("fucina");

test "first program" {
    const alloc = std.testing.allocator;
    var ctx: fucina.ExecContext = undefined;
    ctx.init(alloc);
    defer ctx.deinit();

    // x: [batch=1, in=2], w: [in=2, out=1]
    var x = try fucina.Tensor(.{ .batch, .in }).variable(&ctx, try ctx.fromSlice(.f32, &.{ 1, 2 }, &.{ 2, 3 }));
    defer x.deinit();
    var w = try fucina.Tensor(.{ .in, .out }).variable(&ctx, try ctx.fromSlice(.f32, &.{ 2, 1 }, &.{ 4, 5 }));
    defer w.deinit();

    var y = try x.dot(&ctx, &w, .in); // contract .in => [batch, out]
    defer y.deinit();
    var loss = try y.sumAll(&ctx);
    defer loss.deinit();

    try loss.backward(&ctx);
    var gx = (try x.grad(&ctx)).?; // dloss/dx = w^T = [4, 5]
    defer gx.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 23.0), try loss.item(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), (try gx.dataConst())[0], 1e-6);
}
```

BLAS/GPU configuration passes through `b.dependency` options
(`.blas = .none`, `.backend = .scalar`, ...) and the modules carry their
own link inputs, so the defaults work with no extra build steps. The API is
pre-1.0 and will change: pin the tag (or a commit) you fetch. Details,
option reference, and the vendoring fallback:
[REFERENCE §2.5](https://matteo-grella.github.io/fucina/docs/reference/02-toolchain-build-and-project-wiring/#25-consuming-fucina-from-another-project).

Build with `-Doptimize=ReleaseFast` whenever speed matters (Debug is 10–50x
slower) — that applies to your application exactly as to the in-tree
binaries. **Builds are tuned to the machine that compiles them.** Without
`-Dtarget`, Zig targets the host CPU with its full feature set — as if
`-march=native` were always on — and Fucina's kernels specialize at compile
time (NEON/dotprod arms on Apple Silicon, AVX2/AVX-VNNI on modern x86;
unused arms are not in the binary). Two rules follow: run the binary on the
machine you built it on, and if you must cross-compile, pass `-Dcpu` as
well (e.g. `-Dtarget=x86_64-linux -Dcpu=x86_64_v3`) — a bare `-Dtarget`
gets that architecture's *baseline* features and silently loses the fast
kernels.

## What the library enables today

With the tensor core in place, Fucina grows and gets tested through real
applications, so the runtime and the things built on it develop side by
side. Every family below is ordinary consumer code of the same public
`fucina`/`fucina_models` surface wired in Getting started, split across
two homes: `examples/` holds single-file teaching programs meant to be
read and copied, and `apps/` holds the product- and port-shaped programs
(multi-file, own tests, shims, goldens, real CLI surfaces). Both double
as a corpus of real usage: open any `main.zig` and read how the library
is actually driven. And every family is validated against its
reference implementation, a discipline that is the core of the project:
token-ID-exact tokenizers vs `llama-tokenize`, logit-parity oracles vs
llama.cpp, byte-exact quantization encoders vs ggml, byte-identical GGUF
re-emit. Each family's folder carries its own README
with copy-paste commands; [docs/RUNNING-MODELS.md](docs/RUNNING-MODELS.md)
is the index — verified weight downloads and licenses, plus the machinery
shared across runners (expert streaming, GPU offload, global knobs).

Run one:

```sh
git clone https://github.com/matteo-grella/fucina
cd fucina

# Grab a small model. `hf` is the Hugging Face CLI
# (pip install -U huggingface_hub; formerly `huggingface-cli`); or just
# download the GGUF from your browser into models/.
mkdir -p models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models

# Talk to it
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "What is the capital of France?" --no-think

# Or serve it to any OpenAI client (chat completions + responses, SSE
# streaming, JSON-schema constrained output with -Dllguidance=true)
zig build lmserve -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --port 8080

# No model files needed to verify the toolchain:
zig build test
```

Build options (`-Dbackend`, `-Dblas`, `-Dmax-threads`, `-Dgpu=metal`, …)
are documented in
[the manual's toolchain chapter](https://matteo-grella.github.io/fucina/docs/reference/02-toolchain-build-and-project-wiring/).

### Language models

| Family | What it is |
| --- | --- |
| **[Qwen3](apps/qwen3/README.md)** dense (0.6B–8B) + MoE (30B-A3B, 235B-A22B) | chat / REPL / raw generation, lossless speculative decoding, batch-N multi-conversation decode, JSON-schema/grammar constrained output |
| **DeepSeek V2/V3** (MLA, via [`zig build run`](docs/RUNNING-MODELS.md)) | multi-head latent attention with the compressed KV cache as the default; covers V2-Lite, Moonlight-16B-A3B, and GLM-5.2 (`glm-dsa`) checkpoints |
| **GLM-4.5** family (MoE, via [`zig build run`](docs/RUNNING-MODELS.md)) | V3-style trunk plus the model's own `nextn` layer for lossless multi-token-prediction drafting |
| **[DeepSeek V4 Flash](apps/deepseek4/README.md)** 284B-A13B | hyper-connections trunk with native MTP speculative decoding; the 164.6 GB Q4K release decodes on a 64 GB machine |
| **[Gemma 4](examples/gemma4/README.md)** 26B-A4B (MoE) | chat / REPL / speculative decoding, JSON-schema/grammar constrained output |
| **[Qwen3.5](examples/qwen35/README.md)** 0.8B | hybrid Gated-DeltaNet architecture (conv + delta scan + gated attention) |
| **[DiffusionGemma](apps/diffusion_gemma/README.md)** 26B-A4B | block text-diffusion decoding on the Gemma backbone |
| **Inkling** 975B-A41B (MoE, via [`zig build run`](docs/RUNNING-MODELS.md)) | hybrid local/global-attention multimodal decoder: text chat plus image and audio input towers |
| **[nanochat](apps/nanochat/README.md)** | karpathy/nanochat ported whole: BPE tokenizer training, GPT pretraining, SFT, bits-per-byte eval, and chat — trained from scratch on CPU |

MoE models bigger than RAM are first-class: `--moe-stream` keeps only the
dense weights resident and pages routed experts from disk through a
pinned-set + LRU tier, bit-identical to the resident path — that is how the
142 GB Qwen3-235B and the 164.6 GB V4 Flash decode on a 64 GB machine.

### Speech, vision, and audio

| Family | What it is |
| --- | --- |
| **[Parakeet](apps/parakeet/README.md)** (NVIDIA NeMo FastConformer) | speech-to-text: offline, streaming, and live microphone |
| **[OmniVoice](apps/omnivoice/README.md)** | MaskGIT text-to-speech with voice cloning (Higgs Audio v2 codec included) |
| **[LocateAnything-3B](apps/locate_anything/README.md)** | NVIDIA's open-vocabulary detection VLM: text-prompted labeled boxes, byte-compatible with the reference CLI |
| **[facedetect](apps/facedetect/README.md)** (insightface buffalo_l) | face detection, recognition, gender/age, anti-spoofing, and dense landmarks |
| **[Neural Amp Modeler](apps/nam/README.md)** | `.nam` guitar-amp profiles: run, train, export, live amp simulation |

These applications will eventually graduate into their own repositories.
The known debt of the in-tree phase is that generic operations accumulate
inside the apps — resamplers, spectrograms, reference-parity image
resizing — and graduation starts with an audit of which of those are
really tensor ops that belong in the core. Research experiments that lack
a reference oracle live on `research/*` branches rather than `main` —
currently `research/nla`, a natural-language autoencoder study
(text→vector→text on a Qwen3 GGUF) built on the trainer's hidden-state
seams.

## Performance

Measured, not asserted: the protocol is paired same-machine runs against a
reference implementation — same GGUF, same thread count, CPU-only both
sides — with losses recorded as plainly as wins, each number carrying its
hardware, protocol, and caveats. The short version of the dated record
(snapshot 2026-07-04): on Apple M1 Max, of 236 paired sweep cells across
Qwen3 dense (0.6B/1.7B), Qwen3.5, the 30B MoE, and Gemma-26B, Fucina is
faster in 221 and at parity in 13 — dense prefill geomeans 1.18–1.81x per
format, large MoE prefill up to ~2x. On an x86 Raptor Lake box
(AVX2+VNNI), Fucina wins all dense quantized formats (medians 1.32–1.95x)
while llama.cpp decisively wins MoE small-batch prefill. The full record —
the losing cells, the caveats, and the reproduction commands
(`tools/fetch_refs.sh`, `tools/bench_gate.py`) — is
[docs/BENCHMARK.md](docs/BENCHMARK.md).

On top of raw speed there is lossless, draft-model-free speculative
decoding (up to 2.3x on retrieval-structured tasks, never-a-loss cost
gate; [docs/SPECULATIVE.md](docs/SPECULATIVE.md)) and batch-N multi-stream
decode (3.2x aggregate throughput at 8 streams). Structured output is
built in: a pluggable logit-processor seam on the shared sampler, with
JSON-schema/regex/Lark constrained decoding through the vendored
[llguidance](https://github.com/guidance-ai/llguidance) engine (opt-in
`-Dllguidance=true`; composes with speculative decoding —
[docs/CONSTRAINED-DECODING.md](docs/CONSTRAINED-DECODING.md)).

## Training

The runtime trains as well as it infers: an eager autograd engine with exec
scopes, activation checkpointing, deterministic dropout (counter-based RNG),
and SGD/AdamW/Muon/APOLLO optimizers golden-parity-tested against their
reference implementations. LoRA fine-tuning of a quantized Qwen3 GGUF runs
end-to-end on CPU (~932 ms/step, 38.2 tok/s supervised throughput for
Qwen3-0.6B-Q4_K_S on an M1 Max), and the loop closes: fine-tune → merge →
quantize → the exported GGUF loads and answers in llama.cpp.

```sh
zig build finetune -Doptimize=ReleaseFast -- --steps 30
```

Gradient-free training is a first-class alternative: `fucina.es` implements
evolution strategies at scale (arXiv:2509.24372) — seed-regenerated noise,
forward passes only, algebra cross-checked bitwise against the reference —
and `zig build es-finetune` fine-tunes the same GGUF with it, LoRA-only or
full-parameter, under rule-based (R1-style) or loss-based rewards. Because
autograd, KV-cache plumbing, and serving live in one runtime, the same
trainer also implements **Cartridges** (arXiv:2506.06266): a corpus is
compressed into a trained KV prefix by in-process self-study distillation
(`zig build cartridge`; [docs/CARTRIDGES.md](docs/CARTRIDGES.md)).

[docs/TRAINING.md](docs/TRAINING.md) is the full guide, including how the
gradients were verified (PyTorch goldens, finite differences, and a
real-model audit) and its open issues.

## Design

Fucina is deliberately **eager and local**: no global graph object, no fusion
pass, no compiler layer. The execution context validates shapes once, then
dispatches to small unchecked, allocation-free backend kernels selected at
build time. Transient memory goes through a thread-safe reusable buffer pool
with bucket-rounded buffer allocation for small temporaries — the rationale
(and why it beats an arena here) is in
[docs/MEMORY-MODEL.md](docs/MEMORY-MODEL.md). The intended growth path is
model-specific sessions with semantic weight binding and preallocated
buffers, not a generic ggml-like graph.
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) maps the whole tree.

That choice has a known price, paid deliberately. Whole-graph machinery —
fusion passes, static device memory planning, captured launch sequences — is
what mainstream GPU inference stacks are built on, so an eager runtime can
use a GPU only at the op seam (which is what the Metal and CUDA offloads do)
and will never chase TensorRT-class GPU throughput. On CPU the ledger reads
the other way: per-op dispatch costs next to nothing, kernels specialize at
compile time instead of fusing at runtime, and the paired benchmarks are run
against a graph executor. CPU-first is the design point, not a stage on the
way to somewhere else.

## Documentation

**[The manual](https://matteo-grella.github.io/fucina/docs/)** renders all
of it — reference, guides, course, and per-example pages — with search.
[docs/README.md](docs/README.md) is the full source index; the entry
points:

| Doc | Contents |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the actual source layout, layer by layer — start here |
| [docs/reference/](docs/reference/00-index.md) | the detailed API reference, one chapter per file: the full public surface, exact semantics, machine-verified snippets |
| [docs/RUNNING-MODELS.md](docs/RUNNING-MODELS.md) | the model index: verified weight downloads + licenses, the example-to-README map, shared runner machinery |
| [docs/course/](docs/course/README.md) | *Forging Deep Learning in Zig* — a book-length course that teaches Zig and deep learning together by rebuilding this library's journey, from a dtype enum to a live guitar amp and chatting language models; per-chapter video scripts included |

## Status and scope

Honest expectations:

- **CPU-first, two ISAs.** Tuned on Apple Silicon (aarch64 NEON/sdot) and
  x86-64 (AVX2/AVX-VNNI); a scalar reference backend covers everything else.
  The Metal offload accelerates specific GEMM shapes on macOS; it is not a
  general GPU runtime. The CUDA sibling (`-Dgpu=cuda`, Linux/NVIDIA) plugs
  into the same seam with zero build-time SDK dependency: f32/f16 GEMM via
  dlopen'd cuBLAS, quantized dense + MoE prefill and fused prefill attention
  via vendored PTX kernels, and an opt-in decode GEMV.
- **The API is not stable.** This is a young codebase published in the open,
  not a 1.0 library. The 0.x tags are pins for `zig fetch`, not a semver
  promise. Expect churn.
- **Model weights are not included.** Each model family carries its own
  license (Qwen: Apache-2.0; Gemma: Google's Gemma Terms of Use; Parakeet:
  CC-BY-4.0; OmniVoice weights: CC-BY-NC). `docs/RUNNING-MODELS.md` notes the
  terms next to each download.
- **Benchmarks age.** llama.cpp moves fast; the dated records in
  `docs/BENCHMARK.md` are snapshots, not eternal claims.

## Origins

Fucina grew out of autograd concepts I first explored in Go with
[spaGO](https://github.com/nlpodyssey/spago) — above all the idea that the
graph should be **implicit in the values themselves**: no graph object, no
tape, no persistent engine. Each result carries a pointer to the operation
that produced it, and `backward()` discovers the topology by walking those
pointers. spaGO executed that idea the Go way: one goroutine per node,
each blocking until its gradient contributions arrived, the runtime
scheduler absorbing the wait. Zig has no goroutines, so Fucina keeps the
idea and rethinks the execution: every node carries an atomic dependency
counter, and its gradient fires only when the counter drains — concurrent,
on a bounded worker pool, no blocked workers (`src/ag/`). (AFAIK) Mainstream
stacks route backward through a central engine over an explicit node graph
or a trace; here the live tensors *are* the graph.

As for the language: I wanted to stay as close to the metal as possible, and — honestly — I was
also looking for a good excuse to finally learn Zig. This project is it.

Development — code and documentation alike — leans on strong assistance
from agentic coding systems, with humans leading the ideas, the testing,
and the debugging, and writing first-hand as well. That stance now has a
motto, thanks to Salvatore Sanfilippo:
["Control the ideas, not the code"](https://antirez.com/news/169).
I also wanted to gain practical experience coding with AI, and so far I
have found that the best results come from a review loop where humans and
multiple frontier models critique one another’s specifications and
implementations, iterating toward a shared consensus.

As for what gets built: right now the core — the tensor/autograd
runtime — evolves in lockstep with the examples, growing exactly the ops
each new port demands, and the examples follow a personal criterion
rather than a roadmap: they are the deep-learning architectures I want
to understand deeply, in the spirit of Feynman's "What I cannot create,
I do not understand." The Neural Amp Modeler is the one that outgrew the
criterion — I genuinely enjoy *using* it, which is its own small
achievement: I almost never end up using the things I build.

## Contributing

Contributions are welcome — see `CONTRIBUTING.md`: PRs are
human-owned (coding agents expected, human judgment required), and changes
are tested against the two tracks the project actually regresses on,
correctness and speed, with the commands, machine, and model quant reported.

## Acknowledgments

Fucina exists because others built the road first.

- **ggml / llama.cpp** — Georgi Gerganov and the ggml authors. This project
  would not exist without their work. Fucina is an independent Zig runtime,
  but it speaks formats ggml defined (GGUF, the block-quantization wire
  formats), and several components are direct ports of llama.cpp code: the
  quantization row encoders, the Qwen2 and SentencePiece tokenizers, the
  Unicode classification tables, the SIMD `expf`, and the vendored Metal
  quantized-GEMM kernel. llama.cpp is also the parity oracle and the
  performance yardstick throughout.
- **Ettore Di Giacinto** (mudler, of LocalAI) — the Parakeet ASR family is a
  port of his [parakeet.cpp](https://github.com/mudler/parakeet.cpp) (the
  ready-to-run GGUF weights come from his conversions); the LocateAnything
  open-vocabulary detection example is a port of
  [locate-anything.cpp](https://github.com/mudler/locate-anything.cpp), written
  with **Richard Palethorpe** — its converter/quantizer produce the GGUFs the
  example runs, and it is the parity oracle and CPU yardstick for that port;
  and the facedetect example (SCRFD + ArcFace + GenderAge + anti-spoof +
  dense landmarks) is a port of his
  [face-detect.cpp](https://github.com/mudler/face-detect.cpp), again the
  parity oracle and CPU yardstick, with the buffalo_l GGUFs coming from his
  [conversions](https://huggingface.co/mudler/face-detect-gguf) of the
  [insightface](https://github.com/deepinsight/insightface) models. The
  underlying LocateAnything-3B model is
  [NVIDIA's](https://huggingface.co/nvidia/LocateAnything-3B).
- **Salvatore Sanfilippo** (antirez) — the DeepSeek V4 Flash port follows his
  [ds4](https://github.com/antirez/ds4) inference engine: it is the reference
  for the entire architecture (hyper-connections, compressed sliding
  attention, the FP8/FP4 quantization grids, native MTP), his GGUF
  conversions are the weights the port runs, and his test fixtures (official
  API vectors and local logit goldens) are the validation oracle.
- **Prism ML** — the Ternary Bonsai family is what the `q2_0` ternary
  support serves: the Q2_0 g128 wire format and its reference
  encoder/decoder come from their
  [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp), which is also
  the parity oracle for the Ternary-Bonsai-27B port. The Bonsai weights
  themselves are Apache-2.0 on
  [Hugging Face](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf)
  and are not part of this repository.
- **Thinking Machines Lab** — the Inkling architecture (hybrid local/global
  attention with a banded content-dependent relative-position bias in place of
  RoPE, per-layer short causal convolutions, a fine-grained MoE whose shared
  experts act as routing-softmax sinks, and hierarchical-patch vision plus
  dMel audio towers); the port follows the open weights at
  [thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling).
  **Daniel Han** (Unsloth) — his
  [llama.cpp Inkling architecture PR](https://github.com/ggml-org/llama.cpp/pull/25731)
  is the source of truth for the GGUF layout and graph, and his
  [GGUF conversions](https://huggingface.co/unsloth/inkling-GGUF) provide the
  converter and the parity oracle the port validates against.
- **JustVugg's [colibri](https://github.com/JustVugg/colibri)** — the
  out-of-core MoE expert streaming (run mixture models much bigger than RAM
  by paging routed experts from disk through a pinned-set + LRU tier, with a
  persistent usage histogram and router-lookahead prefetch) was inspired by
  colibri's design; Fucina's implementation is independent, streaming ggml
  quants over the fused kernels.
- **Apple MLX** — the f32/f16 Metal GEMM is the vendored MLX "steel" kernel.
- **guidance-ai / Microsoft** — the vendored
  [llguidance](https://github.com/guidance-ai/llguidance) engine powers
  grammar/JSON-schema constrained decoding.
- **ServeurpersoCom** — the OmniVoice TTS port follows his
  [omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp), which
  also provided the codec porting groundwork (Higgs Audio v2 / HuBERT / DAC).
- **Steven Atkinson** — NeuralAmpModelerCore and neural-amp-modeler, the
  reference for the entire NAM example.
- **ZINC** — the byte-level BPE tokenizer core was adapted from the ZINC Zig
  inference engine.
- **ZML** — the tagged-tensor approach (axis tags carried in the type,
  operands aligned by name) was inspired by [ZML](https://github.com/zml/zml).
- **David Reid** (miniaudio), **Keller Jordan** (Muon), **Sebastiano Vigna**
  (splitmix64), **musl libc** and **ARM optimized-routines** (scalar/SIMD
  math lineage), **k2-fsa/Xiaomi** (OmniVoice), **NVIDIA NeMo** (Parakeet),
  **Boson AI** (Higgs Audio v2), **Meta** (HuBERT), **Descript** (DAC),
  **kitft** (natural-language autoencoders).

The complete inventory — what is vendored, what is ported, what is only a
parity reference, and under which license — is in
[docs/THIRD-PARTY-NOTICES.md](docs/THIRD-PARTY-NOTICES.md).

## License

MIT — see `LICENSE`. One documented exception: `tools/gen_optim_goldens.py`
contains material derived from the APOLLO reference implementation and is
covered by its upstream CC-BY-NC-4.0 terms; see `docs/THIRD-PARTY-NOTICES.md`.
