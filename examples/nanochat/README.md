# nanochat (Fucina port)

A from-scratch CPU port of [karpathy/nanochat](https://github.com/karpathy/nanochat)
onto Fucina's tensor/autograd runtime: BPE tokenizer training, GPT pretraining,
supervised fine-tuning, bits-per-byte evaluation, and an interactive chat CLI with a
calculator tool — the full `runs/runcpu.sh` pipeline, in Zig, on CPU.

The port targets the Python reference (nanochat @ `92d63d4`) running on CPU in fp32.
Every stage is validated against reference oracles under a tiered parity ladder
(`docs/PORTING.md` style): bit-exact where structurally possible, tight-f32 tolerance
on the float path.

## Build & run

```sh
zig build nanochat -- <subcommand> [args]      # Debug
zig build nanochat -Doptimize=ReleaseFast -- <subcommand> [args]   # training/inference
```

Subcommands (`nanochat <cmd>`):

| cmd | what it does |
| --- | --- |
| `tok-train`  | train the rustbpe-equivalent BPE tokenizer → `tokenizer.bin` + `token_bytes.bin` |
| `base-train` | pretrain the GPT on framed pretraining docs (grad-accum loop, Muon+AdamW, bpb eval, checkpoint/resume) |
| `sft`        | supervised fine-tune a base checkpoint on the task mixture (masked loss, SFT schedule) |
| `eval-bpb`   | bits-per-byte over a validation split |
| `chat`       | interactive chat / single-prompt generation (KV-cache decode, calculator tool) |

Example (single-prompt chat from a trained checkpoint):

```sh
zig build nanochat -Doptimize=ReleaseFast -- chat --init-from <model.safetensors> \
    --tokenizer <tokenizer.bin> -p "The capital of France is" -t 0 --base
```

## Architecture

The port is **example-local** (no `src/` changes): everything composes from the public
Fucina facade. Model math maps 1:1 onto existing ops — RMSNorm (no-weight), the
half-split RoPE kernel with an inverse-built table (base 100000), grouped causal
attention, `crossEntropyExt`, `gather`/scatter-add embeddings — and the two pieces the
facade didn't already have are written here: the raw-byte BPE tokenizer and nanochat's
MuonAdamW optimizer variant (Polar-Express orthogonalization + NorMuon variance
reduction + cautious weight decay), the latter reusing Fucina's `AdamW` for its six
Adam groups and a batched-`bmm` Newton–Schulz for the Muon groups.

| file | role |
| --- | --- |
| `tokenizer.zig` | raw-byte BPE: `nanochatChunkEnd` pretokenizer (a `qwen2ChunkEnd` variant differing only in the `\p{N}{1,2}` digit arm), rustbpe-equivalent trainer, tiktoken-style encoder, `tokenizer.bin`/`token_bytes.bin` I/O |
| `model.zig` | GPT: config, params, forward (+ per-layer `Trace`), loss/`lossNone`, and a KV-cached `forwardStep`/`Cache` decode path |
| `optim.zig` | `MuonAdamW` — 6 reused `fucina.optim.AdamW` groups + custom Muon over 4 shape-groups; base/SFT schedules |
| `data.zig` | framed-doc (`NCDOC`) + JSONL readers, `renderConversation`, base BOS-bestfit loader, SFT bestfit-pad loader |
| `train.zig` | the base-train + SFT loops, hyperparameter derivation, bpb eval, checkpoint save/resume |
| `chat.zig` | inference engine, temperature/top-k sampling, calculator-tool state machine, chat CLI |
| `tools/nanochat_export.py` | one-time reference exporters: tokenizer/`token_bytes`, framed pretraining docs, SFT mixture JSONL, encode/loader parity fixtures |
| `tools/nanochat_dump.py` | reference parity-oracle dumps: init / forward (per-layer) / grad / optimizer-step / loss-trace / greedy-decode / bpb |

## Data & goldens

The reference reads HuggingFace parquet shards (ClimbMix pretraining; SmolTalk/MMLU/GSM8K
for SFT). A one-time Python export (`tools/nanochat_export.py`, run against the reference
repo's venv) converts these into neutral little-endian binary formats that the Zig loaders
consume in the exact reference order, so packing/mixture order stays reproducible. The
binary formats are documented in the port's `FORMATS.md`.

Parity goldens (init/forward/grad/optstep/loss-trace/greedy oracles, trained reference
checkpoints, encode + loader fixtures) live untracked under
`refs/nanochat-goldens/`; the parity tests below skip cleanly when they are absent.

## Tests

The tokenizer/model/optimizer/data/train/chat unit tests run under `zig build test`.
Reference-parity suites are gated on the `NANOCHAT_PARITY` environment variable (the
`OMNIVOICE_PARITY` precedent) and skip when unset or when goldens are missing:

```sh
NANOCHAT_PARITY=1 zig build test        # run the parity gates
zig build test                          # always-on unit tests only
```

`FUCINA_TEST_VERBOSE=1` additionally prints the always-on tests' measured
margins (finite-diff relerrs, resume-determinism param count) on passing
runs; by default passing tests are silent so `zig build test` stderr stays
clean.

Parity coverage: tokenizer encode (token-ID-exact) + trainer (merge list byte-identical
to rustbpe); base/SFT dataloader batches byte-identical to the reference; model forward
(per-layer intermediates ≤ 1e-5 rel), grad, and KV-cache-vs-full self-consistency;
optimizer single-step + 10-step vs reference; base/SFT loss-trace within a drift budget;
greedy decode token-exact vs a trained reference checkpoint.

## Getting the data

One-time setup: the Zig pipeline consumes only the exported files; the reference
repo and its Python venv exist just to produce them.

**1. Pin the references** (cloned under the gitignored `refs/`, never vendored):

```sh
tools/fetch_refs.sh nanochat rustbpe
```

checks out `refs/nanochat` @ `92d63d4` and `refs/rustbpe` @ `ddf848f`.

**2. Reference venv + cache.** Set up the reference's own `uv` venv per its README
(`refs/nanochat/.venv/`; torch CPU, tiktoken, rustbpe, pyarrow, safetensors), then
let the reference pull its data and train its tokenizer into the cache
(`NANOCHAT_BASE_DIR`, default `~/.cache/nanochat`):

```sh
cd refs/nanochat
.venv/bin/python -m nanochat.dataset -n 8                      # ClimbMix parquet shards
.venv/bin/python -m scripts.tok_train --max-chars=2000000000   # tokenizer.pkl + token_bytes.pt
```

The SFT task datasets (SmolTalk/MMLU/GSM8K) are downloaded from HuggingFace by the
reference `tasks/` modules the first time an SFT export runs.

**3. Export.** Run `tools/nanochat_export.py` with the venv python and the
reference repo on `PYTHONPATH`; keep outputs under `refs/nanochat-goldens/` (the
untracked goldens dir — `chat` defaults its `--tokenizer` there). From the repo
root:

```sh
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
export PYTHONPATH=$PWD/refs/nanochat
py=refs/nanochat/.venv/bin/python
ex=examples/nanochat/tools/nanochat_export.py
out=refs/nanochat-goldens

$py $ex export-tokenizer --out $out                                              # tokenizer.bin + token_bytes.bin + merges.txt
$py $ex export-train-text --out $out/train_text_small.bin --max-chars 3000000    # NCTXT_01 tok-train corpus
$py $ex export-docs --split train --out $out/base_train_small.bin --max-docs 30000  # NCDOC_01 (+ .idx.json)
$py $ex export-docs --split val   --out $out/base_val.bin                           # NCDOC_01 val split
$py $ex export-sft-mixture --split train --out $out/sft_mixture_train.jsonl         # add --max-convs N to cap
$py $ex export-sft-mixture --split val   --out $out/sft_mixture_val.jsonl --max-convs 2000
```

| file | format | consumed by |
| --- | --- | --- |
| `tokenizer.bin`, `token_bytes.bin` | `NCTOKz01` / `NCTKB_01` | every subcommand (`--tokenizer`) |
| `train_text_small.bin` | `NCTXT_01` | `tok-train --input` |
| `base_train_small.bin` (+ `.idx.json`), `base_val.bin` | `NCDOC_01` | `base-train --data/--val-data`, `eval-bpb --data` |
| `sft_mixture_{train,val}.jsonl` | JSONL | `sft --mixture/--val-mixture` |

The JSONL materializes the reference's deterministic mixture order (the
`random.Random(42)` shuffle is baked in Python). Byte layouts are documented at
their readers/writers in `tokenizer.zig` and `data.zig`; any corpus packed into
those layouts works — the exporter matters when reference-order reproducibility
is the goal. The parity-oracle exports (encode/loader fixtures,
model/optimizer dumps via `nanochat_dump.py`) are covered by the regen recipe in
[`goldens/README.md`](goldens/README.md).

## Quick training run

Smallest end-to-end pass — small tokenizer, tiny model, enough steps to watch the
loss move; artifacts land under `/tmp/nc-*`. Use `-Doptimize=ReleaseFast` for
anything that trains (shared build discipline:
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md)).

1. Tokenizer — 1024-token vocab on the small exported corpus (the goldens'
trainer-gate config):

```sh
zig build nanochat -Doptimize=ReleaseFast -- tok-train \
    --input refs/nanochat-goldens/train_text_small.bin --vocab 1024 --out /tmp/nc-tok
```

Prints the doc/vocab counts, then `trained N merges (n_vocab 1024) in Xs`, and
writes `/tmp/nc-tok/tokenizer.bin` + `token_bytes.bin`.

2. Base pretraining, smallest sensible config (depth 2 ⇒ model_dim 128,
grad_accum 1; `--total-batch-size` must be a multiple of
`device_batch_size × max_seq_len`):

```sh
zig build nanochat -Doptimize=ReleaseFast -- base-train \
    --data refs/nanochat-goldens/base_train_small.bin \
    --tokenizer /tmp/nc-tok/tokenizer.bin --out /tmp/nc-base \
    --depth 2 --max-seq-len 256 --device-batch-size 8 --total-batch-size 2048 \
    --num-iterations 50
```

Expected output: two header lines with the derived hyperparameters
(`model_dim=128 num_heads=2 …`, `scaling_params=… grad_accum=1 …`), one
`step NNNNN/00050 | loss: … | lrm: … | dt: …ms` line per step (the per-step `dt`
tells you what a longer run costs on your machine), and at the final step greedy
`sample:` previews plus `checkpoint written to /tmp/nc-base`. The checkpoint
directory holds `model.safetensors` + `optimizer.fucina` + `trainer_state.json`
(+ a `prev/` rotation of the previous complete save); interrupt and continue with
`--resume /tmp/nc-base` (drop `--init-from`). The defaults are the d6 acceptance
config (`--depth 6 --max-seq-len 512 --device-batch-size 32 --total-batch-size
16384 --num-iterations 5000`); `--total-batch-size 0` / `--num-iterations 0`
switch to the reference's auto Power-Lines batch / scaling-law horizon. Add
`--val-data refs/nanochat-goldens/base_val.bin` for periodic `val bpb` lines
(`--eval-every`, default 500).

3. Chat with the checkpoint (model config is inferred from the safetensors
shapes; 50 tiny-model steps prove the pipeline, not a conversationalist — expect
near-gibberish continuations):

```sh
zig build nanochat -Doptimize=ReleaseFast -- chat -i /tmp/nc-base \
    --tokenizer /tmp/nc-tok/tokenizer.bin -p "The capital of France is" -t 0 --base
```

`--base` completes the raw prompt (requires `-p`); drop `-p` for the interactive
REPL, drop `--base` for the chat-protocol wrapping (sensible only after `sft`).
Sampling knobs: `-t/--temperature` (default 0.6), `-k/--top-k` (50),
`--max-tokens` (256), `--seed` (42). Without `--tokenizer`, chat defaults to
`refs/nanochat-goldens/tokenizer.bin`.

Onward: `sft --init-from /tmp/nc-base/model.safetensors --mixture
refs/nanochat-goldens/sft_mixture_train.jsonl --tokenizer /tmp/nc-tok/tokenizer.bin
--out /tmp/nc-sft` fine-tunes on the task mixture (warm-starting the optimizer
from the sibling `optimizer.fucina`), and `eval-bpb --init-from
<model.safetensors> --data refs/nanochat-goldens/base_val.bin --tokenizer
<tokenizer.bin>` reports bits-per-byte. Both take the model geometry from their
CLI flags, not the checkpoint — pass the same `--depth/--max-seq-len` you trained
with (here `--depth 2 --max-seq-len 256`); `--dtype` must likewise match the
checkpoint on `--init-from`/`--resume`.
