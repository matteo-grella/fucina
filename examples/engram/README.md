# Engram — conditional n-gram memory grafted onto Qwen3

Attach DeepSeek's Engram (arXiv 2601.07372) — hashed n-gram lookup tables
gated into the residual stream — to a FROZEN Qwen3 GGUF and train only the
graft (optionally plus LoRA adapters).
[docs/ENGRAM.md](../../docs/ENGRAM.md) is the design record for the
underlying `src/llm/engram.zig` module (reference semantics, parity gates,
graft mode); the API surface is `docs/REFERENCE.md` §13.11. This example is
the experiment driver: it wires the module into the qwen3 trainer and
measures whether the memory helps.

Because the graft's value projection is zero-initialized
(`graft_zero_init`), the grafted model is bitwise identical to the bare
model at step 0 — that identity is a runnable gate (`--equiv`), not an
assumption.

## Getting the model

Weights are not part of the repository. Any Qwen3 GGUF the `qwen3` runner
loads works (`--model` takes a plain path); the default path is
`models/Qwen3-0.6B-f16.gguf`. Download sources are in the weights table of
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md#getting-the-weights):

```sh
mkdir -p models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models
```

The Q8_0 download works as-is (pass `--model models/Qwen3-0.6B-Q8_0.gguf`
to the commands below): the trainer keeps frozen weight memory quantized
and trains only the graft (and LoRA) in f32. Dense Qwen3 only — MoE bases
are rejected.

If you want the default f16 file and your source only ships bf16, transcode
one locally:

```sh
zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf <src>.gguf --out models/Qwen3-0.6B-f16.gguf --dtype f16
```

## CLI

```sh
# Gate: the zero-init graft must leave the frozen model BITWISE unchanged
# (every logit compared element-for-element; exits nonzero on any diff)
zig build engram -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf --equiv

# Train the graft: continued-pretraining next-token CE over --corpus chunks;
# the trunk stays frozen (--lora N adds trainable q/v adapters)
zig build engram -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
    --corpus docs --train --steps 200 --save graft.safetensors

# Eval a saved graft: held-out CE, bare arm vs grafted arm
zig build engram -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
    --corpus docs --eval --load graft.safetensors
```

`--corpus` is one file or a directory (its top-level `.md`/`.txt` files,
sorted), tokenized into a single flat stream and split into `--chunk`-token
chunks; every 8th chunk is held out, and the stream must cover at least 4
chunks. During training, held-out CE (bare arm vs grafted arm) prints every
`--eval-every` steps.

The example commands point `--corpus` at the repository's own `docs/`
directory: a fresh clone needs no dataset download (its top-level `.md`
files alone exceed the 4-chunk minimum; subdirectories such as
`docs/course/` are not read). Any other plain-text file or directory
works the same way — point `--corpus` at whatever text the graft should
memorize.

| flag | meaning |
| --- | --- |
| `--model <gguf>` | frozen Qwen3 GGUF, default `models/Qwen3-0.6B-f16.gguf` |
| `--corpus <file\|dir>` | training/eval text (required for `--train`/`--eval`) |
| `--equiv` / `--train` / `--eval` | mode (pass one) |
| `--steps N` | training steps, default 200 |
| `--chunk N` | tokens per chunk, default 256 |
| `--lr F` | AdamW learning rate, default 1e-3 |
| `--lora R` | LoRA rank on q/v (alpha `2R`); 0 (default) = graft-only |
| `--seed N` | seed for multipliers/init/probe sampling, default 7 |
| `--save F` / `--load F` | write (after training) / read (before the mode runs) the graft state dict |
| `--eval-every N` | held-out eval interval during training, default 25 |
| `--eval-chunks N` | held-out chunks scored per eval, default 8 |
| `--layers a,b` | Engram layer ids; default `1,num_layers/2` (the reference early+middle placement) |
| `--table-vocab N` | per-head table size target per n-gram order (head primes search upward from it), default 100000 |
| `--n-embed N` | retrieved embedding width per n-gram order, default 256 |
| `--heads N` | hash heads per n-gram order, default 4 |
| `--gate-bias F` | initial gate bias, default 0 |
| `--no-engram` | control arm: train/eval without attaching the graft (clean LoRA-only baseline) |
| `--probes N` | after training/eval: N verbatim-recall probes per source (see below) |

Geometry fixed by the driver: two n-gram orders (`max_ngram_size = 3`),
plain residual stream (`hc_mult = 1`), ShortConv kernel size 4, pad id 0.
Multipliers and primes derive from `--seed`; the saved state dict is
self-describing (multipliers included).

Probes (`--probes N`) score teacher-forced verbatim recall: 16-token
targets after 32-token corpus prefixes, sampled deterministically from
train spans and from held-out spans, each scored with the engram detached
and attached (CE + greedy exact-match rate). Recall is the memory's actual
job; CE on random text under-rewards it. All measurements are
teacher-forced (held-out CE + next-token accuracy) — the go/no-go signal
for the graft experiment; this driver has no serving integration.

```sh
# Recall probes (append to --train or --eval): N spans per source,
# scored with the engram detached and attached
zig build engram -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-Q8_0.gguf \
    --corpus docs --eval --load graft.safetensors --probes 8
```

Probes need `--chunk` >= 48 (32-token prefix + 16-token target); with a
smaller chunk they are skipped.

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, `-Dcpu=...` when
cross-compiling), global thread/BLAS knobs, GPU offload, and `-Dllguidance`
constrained-decoding usage are documented once in
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). This driver takes
only the flags listed above.
