# GLM-4.5 family — `zig build glm4moe`

Greedy completion with optional native MTP (multi-token-prediction)
speculative decoding.

This is one of the three DeepSeek-family runners — the siblings are
[DeepSeek V2/V3 (`deepseek2`)](../deepseek2/README.md) and
[DeepSeek V4 Flash (`deepseek4`)](../deepseek4/README.md). All three share
the streamed-expert machinery and accept `--moe-stream`/`--moe-cache-mb`
([Streaming MoE experts from disk](../../docs/RUNNING-MODELS.md#streaming-moe-experts-from-disk-out-of-core-models-bigger-than-ram)).

V3-style MoE trunk plus the model's own `nextn` multi-token-prediction
layer: `--mtp[=depth]` drafts with the MTP head and verifies with one
batched trunk step — only greedy-matching prefixes commit, so output is
lossless (byte-identical to plain greedy). Measured 2.29 tokens per forward
at depth 2 on GLM-4.5-Air Q6_K streamed on a 64 GB machine. Depth caps
at 2: the verify batch is depth+1 rows, and the m ≥ 4 x4-packed kernels
legally drift from the S=1 numerics, which would break losslessness.

## Getting the model

Model weights are not part of this repository. The command below uses the
GLM-4.5-Air Q6_K split GGUF under `models/glm45-air/`; point the runner at
part 1 — llama.cpp split GGUFs (`-00001-of-0000N`) load transparently.

## Run

```sh
zig build glm4moe -Doptimize=ReleaseFast -- \
  models/glm45-air/GLM-4.5-Air-Q6_K-00001-of-00002.gguf \
  --prompt "..." --gen 64 --mtp --moe-stream --moe-cache-mb=20480
```

Prints load/prefill/decode timings (decode includes tokens per forward),
the generated token ids and the completion text. With `--mtp` it also
prints the draft-acceptance rate and a feed hit-rate diagnostic (the MTP
head's next-next-token hit rate on known history — separates a broken MTP
forward, near 0%, from a broken draft/verify loop, healthy 30–60%).
The canonical GLM `[gMASK]<sop>` opening is added automatically; on a model
without a `nextn` layer `--mtp` is ignored with a notice.

Flags (first positional argument = model GGUF, required):

| flag | meaning |
| --- | --- |
| `--prompt "..."` / `--prompt=...` | prompt text (default `The capital of France is`) |
| `--gen N` / `--gen=N` | greedy tokens to generate, default 32 |
| `--mtp` / `--mtp=depth` | native MTP speculative decoding; bare `--mtp` = depth 2, values above 2 clamp to 2 |
| `--moe-stream` / `--moe-cache-mb=N` | streamed experts — see *Shared knobs* |

## Shared knobs

MoE expert streaming, GPU offload (`-Dgpu=metal`/`-Dgpu=cuda`), global
thread/BLAS knobs and the ReleaseFast/`-Dcpu` build discipline are shared
machinery — see [docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md).
Of the streaming knob set this runner parses only `--moe-stream` and
`--moe-cache-mb`.
