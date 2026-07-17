# DeepSeek V2/V3 (MLA) — `zig build deepseek2`

Greedy completion over the multi-head-latent-attention + MoE forward,
with optional DeepSeek Sparse Attention (DSA) on checkpoints that ship the
lightning indexer.

This is one of the three DeepSeek-family runners — the siblings are
[GLM-4.5 (`glm4moe`)](../glm4moe/README.md) and
[DeepSeek V4 Flash (`deepseek4`)](../deepseek4/README.md). All three share
the streamed-expert machinery and accept `--moe-stream`/`--moe-cache-mb`
([Streaming MoE experts from disk](../../docs/RUNNING-MODELS.md#streaming-moe-experts-from-disk-out-of-core-models-bigger-than-ram)).

Multi-head latent attention with the compressed KV cache as the default
(576 floats per token per layer, 8.9× smaller than reconstructed heads;
`--mla=full` selects the reconstructing path, byte-identical output) and
weight absorption folding `kv_b` into the query/value sides. Covers V2-Lite
(softmax router) and V3-style checkpoints such as Moonlight-16B-A3B
(sigmoid no-aux router, q-LoRA, MLA-native GGUF layout), plus GLM `glm-dsa`
checkpoints (MLA-native attention + V3 sigmoid routing under their own
metadata prefix). When the vocab defines `[gMASK]`/`<sop>` the prompt opens
with them instead of BOS — GLM trunks degenerate without that opening.

## Getting the model

Model weights are not part of this repository, and no download source is
pinned for this family — the GGUF is user-supplied. The commands below use
a GGUF conversion of DeepSeek-V2-Lite-Chat under `models/`; any
V2/V3-family GGUF (V2-Lite, Moonlight-16B-A3B, glm-dsa) works — the runner
takes a plain path. The `hf` CLI download pattern
(`hf download <repo> <file> --local-dir models`) and the per-family weight
table live in
[docs/RUNNING-MODELS.md — Getting the weights](../../docs/RUNNING-MODELS.md#getting-the-weights).

## Run

```sh
zig build deepseek2 -Doptimize=ReleaseFast -- \
  models/DeepSeek-V2-Lite-Chat.Q8_0.gguf --prompt "..." --gen 64
```

Prints load/prefill/decode timings and the greedy completion.

Flags (first positional argument = model GGUF, required):

| flag | meaning |
| --- | --- |
| `--prompt "..."` / `--prompt=...` | prompt text (default `The capital of France is`) |
| `--prompt-file=PATH` | read the prompt from a file (≤ 16 MiB) |
| `--gen N` / `--gen=N` | greedy tokens to generate, default 32 |
| `--ctx=N` | KV capacity override (default `max(2048, prompt+gen+8)`) |
| `--prefill-chunk=N` | batched-prefill chunk size, default 64; `1` restores the sequential S=1 path |
| `--mla=latent\|full` | `latent` (default) decodes from the compressed KV cache; `full` reconstructs heads — byte-identical output |
| `--nll-file=PATH` | teacher-forced NLL/perplexity over a text file, then exit (the dense-vs-DSA quality gate) |
| `--dsa` | load the DSA lightning-indexer tensors (V3.2 / glm-dsa files) and attend sparsely past `indexer_top_k` positions — the trained behavior of those checkpoints |
| `--dsa-top-k=N` | selection-threshold override so the sparse path fires within a short prompt; selection semantics unchanged (combine with `--dsa`) |
| `--index-probe` | decode-time selection-overlap probe across DSA layers; measures the exact path, so mutually exclusive with `--index-share` (implies `--dsa`) |
| `--index-share=N` | cross-layer indexer reuse: every Nth DSA layer computes its selection, the layers between reuse it — approximate by design, calibrate with the probe first (implies `--dsa`) |
| `--moe-experts=N` | inference-time truncation of the routed-expert count; dropped experts are never fetched, gate weights renormalize |
| `--moe-top-p=F` | dynamic expert drop: keep routed experts covering fraction F of the gate mass (deterministic) |
| `--moe-skip-miss=F` | dynamic expert drop: skip sub-threshold-weight experts only when they would cost a disk read (cache-state dependent output) |
| `--moe-stream` / `--moe-cache-mb=N` / `--moe-pilot` | streamed experts — see *Shared knobs* |

## Shared knobs

MoE expert streaming, GPU offload (`-Dgpu=metal`/`-Dgpu=cuda`), global
thread/BLAS knobs and the ReleaseFast/`-Dcpu` build discipline are shared
machinery — see [docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md).
Of the streaming knob set this runner parses `--moe-stream`,
`--moe-cache-mb` and `--moe-pilot`; its expert-drop dials are the
`--moe-experts`/`--moe-top-p`/`--moe-skip-miss` flags above.
