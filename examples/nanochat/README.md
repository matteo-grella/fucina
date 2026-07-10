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
