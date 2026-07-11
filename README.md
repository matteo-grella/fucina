# Fucina

**Fucina** (Italian for *forge*) is a CPU-first tensor/autograd runtime and
LLM inference engine written in pure **Zig 0.16**. Computation is **eager**:
every op executes the moment the model code calls it, on real buffers.
There is no graph to build, plan, or compile first, so what you read is what
runs, in inference and in training alike. Eager does not mean slow: on most
measured CPU shapes Fucina matches or beats llama.cpp (`docs/BENCHMARK.md` keeps
the dated records, losses included). There is no C/C++ build system, no
Python runtime dependency, and no ggml-style graph executor — just Zig
vector kernels, optional CBLAS providers for GEMM, and an optional Metal
offload for large GEMMs on macOS.

## What it looks like

Axis tags and rank are **comptime facts**: `Tensor(.{ .batch, .in })` is a
different type from `Tensor(.{ .in, .batch })`, contraction is by axis
*name*, the result's tag set is computed at compile time, and a misaligned
contraction is a compile error — not a runtime shape crash three layers
deep. The pattern, condensed (see `examples/spirals.zig` and the production
trainer in `src/llm/qwen3/train.zig`):

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

The shape discipline lives in the program text, the way it did in Fortran —
`real A(n,m)` told you the rank before you read a single loop. Here the
signature `x: Tensor(.{ .batch, .in }) -> Tensor(.{ .batch, .class })` lets
you check the algorithm against the math by eye, and Zig's comptime makes it
free: the tags exist only in the type system and compile away entirely.

## Design

Fucina is deliberately **eager and local**: no global graph object, no fusion
pass, no compiler layer. The execution context validates shapes once, then
dispatches to small unchecked, allocation-free backend kernels selected at
build time. Transient memory goes through a thread-safe reusable buffer pool
with bucket-rounded buffer allocation for small temporaries — the rationale
(and why it beats an arena here) is in `docs/MEMORY-MODEL.md`. The intended
growth path is model-specific sessions with semantic weight binding and
preallocated buffers, not a generic ggml-like graph. `docs/ARCHITECTURE.md` maps
the whole tree.

## What runs today

| Family | What it is |
| --- | --- |
| **Qwen3** dense (0.6B–8B) + **Qwen3-30B-A3B** MoE | chat / REPL / raw generation, lossless speculative decoding, batch-N multi-conversation decode, JSON-schema/grammar constrained output |
| **Gemma 4** 26B-A4B (MoE) | chat / REPL / speculative decoding, JSON-schema/grammar constrained output |
| **Qwen3.5** 0.8B | hybrid Gated-DeltaNet architecture (conv + delta scan + gated attention) |
| **DiffusionGemma** 26B-A4B | block text-diffusion decoding on the Gemma backbone |
| **Parakeet** (NVIDIA NeMo FastConformer) | speech-to-text: offline, streaming, and live microphone |
| **OmniVoice** | MaskGIT text-to-speech with voice cloning (Higgs Audio v2 codec included) |
| **Neural Amp Modeler** | `.nam` guitar-amp profiles: run, train, export, live amp simulation |

Every family is validated against its reference implementation, and that
discipline is the core of the project: token-ID-exact tokenizers vs
`llama-tokenize`, logit-parity oracles vs llama.cpp, byte-exact quantization
encoders vs ggml, byte-identical GGUF re-emit. `docs/RUNNING-MODELS.md` has
copy-paste commands and verified download links for every model.

These applications live in `examples/` and each will eventually graduate
into its own repository. Meanwhile they are here for convenience, and not by
accident: with the Tensor core in place, Fucina grows and gets tested through real
applications, so the runtime and the things built on it develop side by
side. Research experiments that lack a reference oracle live on
`research/*` branches rather than `main` — currently `research/nla`, a
natural-language autoencoder study (text→vector→text on a Qwen3 GGUF)
built on the trainer's hidden-state seams.

## Quick start

Requires [Zig 0.16.0](https://ziglang.org/download/) — the toolchain is
pinned; other versions will not build.

```sh
git clone https://github.com/matteo-grella/fucina
cd fucina
zig build test          # unit tests, no model files needed

# grab a small model and talk to it
mkdir -p models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "What is the capital of France?" --no-think
```

Build with `-Doptimize=ReleaseFast` whenever speed matters (Debug is 10–50x
slower). Build options (`-Dbackend`, `-Dblas`, `-Dmax-threads`, `-Dgpu=metal`,
…) are documented in `AGENTS.md`.

**Builds are tuned to the machine that compiles them.** Without `-Dtarget`,
Zig targets the host CPU with its full feature set — as if `-march=native`
were always on — and Fucina's kernels specialize at compile time
(NEON/dotprod arms on Apple Silicon, AVX2/AVX-VNNI on modern x86; unused
arms are not in the binary). Two rules follow: run the binary on the
machine you built it on, and if you must cross-compile, pass `-Dcpu` as
well (e.g. `-Dtarget=x86_64-linux -Dcpu=x86_64_v3`) — a bare `-Dtarget`
gets that architecture's *baseline* features and silently loses the fast
kernels.

## Performance

Measured, not asserted: the protocol is paired same-machine runs against
a reference implementation.

For instance, llama.cpp — same GGUF, same thread
count, CPU-only both sides — and `docs/BENCHMARK.md` records losses as plainly
as wins, each number with its hardware, protocol, and caveats, the whole record a dated snapshot.
The short version of the record (snapshot 2026-07-04): on Apple M1 Max, of 236 paired
sweep cells across Qwen3 dense (0.6B/1.7B), Qwen3.5, the 30B MoE, and Gemma-26B,
Fucina is faster in 221 and at parity in 13 — dense prefill geomeans 1.18–1.81x per format, large
MoE prefill up to ~2x — with two cells on llama.cpp's side: Qwen3.5-0.8B at
pp32 (0.86x; its neighbors pp128 and decode win) and one 30B decode cell
recorded before the stricter prewarmed protocol. The smallest gemma batch
cells await one pristine-machine confirmation, as `docs/BENCHMARK.md` discloses.
On an x86 Raptor Lake box (AVX2+VNNI), Fucina wins all dense quantized
formats (medians 1.32–1.95x) while llama.cpp decisively wins MoE
small-batch prefill.

Every row is reproducible — fetch the pinned reference implementations,
then run the paired gate:

```sh
tools/fetch_refs.sh --build   # references into refs/ (gitignored), llama.cpp built CPU-only
python3 tools/bench_gate.py --models qwen3-0.6b-q6_k --tasks prefill,decode
```

On top of raw speed there is lossless, draft-model-free speculative decoding (up to
2.3x on retrieval-structured tasks, never-a-loss cost gate; `docs/SPECULATIVE.md`)
and batch-N multi-stream decode (3.2x aggregate throughput at 8 streams).
Structured output is built in: a pluggable logit-processor seam on the shared
sampler, with JSON-schema/regex/Lark constrained decoding through the vendored
[llguidance](https://github.com/guidance-ai/llguidance) engine (opt-in
`-Dllguidance=true`; composes with speculative decoding — REFERENCE.md §13.6).

## Training

The runtime trains as well as it infers: an eager autograd engine with exec
scopes, activation checkpointing, deterministic dropout (counter-based RNG),
and SGD/AdamW/Muon/APOLLO optimizers golden-parity-tested against their
reference implementations. LoRA fine-tuning of a quantized Qwen3 GGUF runs
end-to-end on CPU, and the loop closes: fine-tune → merge → quantize →
the exported GGUF loads and answers in llama.cpp.

```sh
zig build finetune -Doptimize=ReleaseFast -- --steps 30
```

Gradient-free training is a first-class alternative: `fucina.es` implements
evolution strategies at scale (arXiv:2509.24372) — seed-regenerated noise,
forward passes only, algebra cross-checked bitwise against the reference —
and `zig build es-finetune` fine-tunes the same GGUF with it, LoRA-only or
full-parameter, under rule-based (R1-style) or loss-based rewards.

`docs/TRAINING.md` is the full guide, including how the gradients were verified
(PyTorch goldens, finite differences, and a real-model audit) and its open
issues.

## Documentation

| Doc | Contents |
| --- | --- |
| `docs/ARCHITECTURE.md` | the actual source layout, layer by layer — start here |
| `docs/REFERENCE.md` | the detailed API reference: the full public surface, exact semantics, machine-verified snippets |
| `docs/RUNNING-MODELS.md` | copy-paste CLI commands + verified weight downloads for every model |
| `docs/BENCHMARK.md` | the measurement protocol and dated Fucina-vs-llama.cpp records, wins and losses |
| `docs/TRAINING.md` | the training guide: autograd, optimizers, LoRA, evolution strategies, checkpoints, gradient verification |
| `docs/MEMORY-MODEL.md` | ownership rules and the buffer-pool-not-arena adjudication |
| `docs/PORTING.md` | the porting method — how every model family here earned its parity claims, written for the next port |
| `docs/SPECULATIVE.md` | design record: lossless draft-model-free speculative decoding |
| `AGENTS.md` | build/test/bench commands, build options, repo map, house rules |
| `docs/THIRD-PARTY-NOTICES.md` | full provenance and license inventory of third-party material |

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
  not a 1.0 library. Expect churn.
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
and the debugging, and writing first-hand as well. I also wanted to gain
practical experience coding with AI, and so far I have found that the best
results come from a review loop where humans and multiple frontier models
critique one another’s specifications and implementations, iterating toward
a shared consensus.

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
parity reference, and under which license — is in `docs/THIRD-PARTY-NOTICES.md`.

## License

MIT — see `LICENSE`. One documented exception: `tools/gen_optim_goldens.py`
contains material derived from the APOLLO reference implementation and is
covered by its upstream CC-BY-NC-4.0 terms; see `docs/THIRD-PARTY-NOTICES.md`.
