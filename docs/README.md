# Fucina documentation

The one index of the project documentation. The API reference lives in
[reference/](reference/00-index.md), one chapter per file; the book-length
course lives in [course/](course/README.md); every guide below carries a
`docs-nav` header naming its sidebar group on the
[published site](https://matteo-grella.github.io/fucina/docs/), and
`zig build doc-check` verifies this table lists exactly those guides.

## Reference and course

| What | Where | Contents |
| --- | --- | --- |
| API reference | [reference/](reference/00-index.md) | the full public surface with exact semantics (ownership, errors, defaults, thread-safety) and machine-verified snippets; chapter files `reference/01-*.md` through `reference/14-*.md` |
| Course | [course/](course/README.md) | *Forging Deep Learning in Zig*: a book-length course that teaches Zig and deep learning together by rebuilding this library, with per-chapter video scripts |

## Guides

| Group | Guide | Contents |
| --- | --- | --- |
| Run & serve | [RUNNING-MODELS.md](RUNNING-MODELS.md) | model and example index: verified weight downloads with license notes, the build-step-to-README map, shared runner machinery (MoE streaming, MTP drafting, constrained decoding, GPU offload, global knobs) |
| Run & serve | [LMSERVER.md](LMSERVER.md) | the lmserve server: OpenAI and Anthropic API mapping tables, the accept-concurrently/generate-sequentially architecture, streaming contracts, the per-model backend matrix |
| Run & serve | [SPECULATIVE.md](SPECULATIVE.md) | design record: lossless draft-model-free speculative decoding, the losslessness proof obligations, verify economics, bench results with caveats |
| Run & serve | [CONSTRAINED-DECODING.md](CONSTRAINED-DECODING.md) | design record: grammar/JSON-schema constrained decoding over the `LogitProcessor` seam and the vendored llguidance engine, speculation composition |
| Run & serve | [CARTRIDGES.md](CARTRIDGES.md) | design record: trained KV-prefix corpus compression (Cartridges), self-study distillation, serving, the prefill-equivalence gate |
| Run & serve | [RUNNER.md](RUNNER.md) | the descriptor runner: Descriptor/Model, the two block styles, recorded-golden correctness status, per-family coverage |
| Training | [TRAINING.md](TRAINING.md) | training guide: tensor lifetimes, exec scopes, optimizers, gradient accumulation, checkpoints, LoRA fine-tuning, mixed precision, evolution strategies |
| Quantization | [PTQTP.md](PTQTP.md) | PTQTP trit-plane post-training quantization: the solver, the packing identity, the quantize-and-run recipe, measured quality ladders |
| Quantization | [TERNARY.md](TERNARY.md) | TQ2_0 ternary weights as a first-class citizen: mul-free kernels, encoders, STE training, ternary-native ES, GGUF interop |
| Memory & compute | [MEMORY-MODEL.md](MEMORY-MODEL.md) | why transient memory is per-tensor deinit plus a buffer pool rather than an arena, with file:line evidence and sharp edges |
| Memory & compute | [GPU-OFFLOAD.md](GPU-OFFLOAD.md) | graphless eager GPU completion design: persistent queues, storage fences, transfer reuse, work gates, Metal/CUDA measurements |
| Memory & compute | [SUBQUADRATIC-ATTENTION.md](SUBQUADRATIC-ATTENTION.md) | research record: the SubQ sparse-attention evaluator behind the runner's attention override, contract, kernel, measured scaling |
| Memory & compute | [ENGRAM.md](ENGRAM.md) | design record: conditional n-gram memory (Engram), hashed lookup tables, graft mode, reference parity |
| Project | [ARCHITECTURE.md](ARCHITECTURE.md) | the current architecture from the actual source layout: layer stack, bands, enforcement, known limitations |
| Project | [PORTING.md](PORTING.md) | the porting method: oracle-first staging, the tiered tolerance policy, the parity ladder, two-way interop proof |
| Project | [DEVELOPMENT.md](DEVELOPMENT.md) | the development method: invariants, the capability inventory, the gate matrix, test layout and CI, API stability rules |
| Project | [BENCHMARK.md](BENCHMARK.md) | benchmark protocol for the GGUF runners plus dated measurement snapshots; read before making perf claims |
| Project | [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) | provenance and licenses of the vendored third-party code |

## Root documents

| Doc | Contents |
| --- | --- |
| [README.md](../README.md) | overview, build, model families, scope |
| [AGENTS.md](../AGENTS.md) | toolchain, build/test/bench commands, build options, repo map, house rules |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | the contribution bar: human-owned PRs, the two regression tracks, reporting requirements |
| [CHANGELOG.md](../CHANGELOG.md) | versioned public-API changes with one-line rewrites; stability tiers |

Per-example getting-started guides live next to each entry point as
examples/&lt;name&gt;/README.md; [RUNNING-MODELS.md](RUNNING-MODELS.md)
maps every build step to its README.
