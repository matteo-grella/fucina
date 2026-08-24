# tools/

Development-surface tooling. Tiers, in order of how load-bearing they are:

- **build gate**: wired into a `zig build` step that CI/pre-merge discipline
  runs; breaking it breaks the build contract.
- **release tool**: run on demand to produce or convert artifacts.
- **golden generator**: one-shot Python that produced recorded values now
  embedded in tests; kept verbatim for provenance (Python is research
  scaffolding only — shipped features self-calibrate in Zig).
- **reference harness**: drives an external reference (torch, llama.cpp,
  upstream C++) for parity or apples-to-apples measurement.
- **rig script**: hardware-specific bring-up (CUDA rig, x86 ISA checks).

| tool | tier | purpose |
|---|---|---|
| `check_import_graph.zig` | build gate | `zig build arch-check`: zero import SCCs, every test file forwarded |
| `check_doc_links.zig` | build gate | `zig build doc-check`: doc-index links exist, README fetch pin matches build.zig.zon |
| `gen_snippet_tests.zig` | build gate | `zig build snippet-check`: every runnable docs/reference snippet compiles and asserts |
| `gen_docs_site.zig` | build gate | stages the MkDocs manual from the repo markdown (docs site is self-derived) |
| `bench_gate.py` / `opbench_gate.py` | build gate | measured-throughput regression gates over the bench binaries |
| `export_gguf.zig` | release tool | GGUF export: the train, export, serve-anywhere loop |
| `convert_ds4_fp4.zig` | release tool | DeepSeek-V4 fp4-expert to tied-PTQTP GGUF converter |
| `convert_shine.py` | release tool | SHINE checkpoint to GGUF converter |
| `replay_experts.zig` | release tool | replay a recorded MoE routing trace through cache policies |
| `bench_subq_decode.zig` / `bench_subq_kernels.zig` / `bench_subq_scaling.zig` / `eval_subq_freerun.zig` | research (compile-gated via bench-check) | SubQ attention evaluator benchmarks and Gate C evaluation |
| `gen_cartridge_goldens.py` / `gen_engram_goldens.py` / `gen_es_goldens.py` / `gen_optim_goldens.py` / `gen_qwen3_train_goldens.py` / `gen_shine_goldens.py` / `gen_shine_train_goldens.py` / `kimi3_goldens.py` | golden generator | produced the recorded values in the matching `*_tests.zig` |
| `gen_unicode_categories.py` | golden generator | produced `src/llm/unicode_categories.zig` |
| `bench_generate_mps.py` / `bench_lora_train_mps.py` / `bench_shine_train_mps.py` | reference harness | torch-MPS apples-to-apples counterparts of the Zig benches |
| `check_es_parity.py` | reference harness | evolution-strategies parity vs the Python reference |
| `torch_opbench.py` / `torch_train_step.py` | reference harness | torch op/train-step timing counterparts |
| `llama_logits.cpp` | reference harness | llama.cpp logit dumper for model parity |
| `export_squad_triples.py` | reference harness | SQuAD to SHINE triple JSONL (research scaffolding) |
| `pocket/` | reference harness | Pocket TTS reference dump + GGUF conversion scripts |
| `ref-patches/` | reference harness | patches applied to external references to expose parity dump points |
| `gen_cuda_ptx.zig` / `gen_cuda_ptx.sh` | rig script | compile vendored CUDA kernels to PTX via NVRTC (CUDA rig) |
| `fetch_refs.sh` / `fetch_voice_models.sh` | rig script | fetch pinned external references / voice model artifacts |

One rig checker lives outside this directory: `src/x86dot_check.zig`
(cross-ISA dot-kernel attestation) is pinned to `src/` by Zig's
module-root rule; its header explains why.

The gates' invocation contract lives in `CONTRIBUTING.md` and
`docs/DEVELOPMENT.md`; everything else documents itself in its header.
