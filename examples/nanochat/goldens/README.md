# nanochat parity goldens

The `NANOCHAT_PARITY=1` test suites validate the Zig port against reference oracles.
The oracle files are **not committed** — they are large and derive deterministically
from the reference. They live untracked under `refs/nanochat-goldens/` (the AGENTS.md
convention for large local captures), and the parity tests skip cleanly when absent.

This file is the provenance + regeneration recipe.

## Provenance

- Reference repo: `karpathy/nanochat` @ `92d63d4` and `karpathy/rustbpe` @ `ddf848f`,
  cloned under `refs/` (`tools/fetch_refs.sh nanochat rustbpe`).
- Reference environment: the nanochat repo's own `uv` venv
  (`refs/nanochat/.venv/`, torch 2.9.1 CPU, tiktoken, rustbpe, pyarrow, safetensors).
- Cache: `NANOCHAT_BASE_DIR=$HOME/.cache/nanochat` (tokenizer, ClimbMix shards, the
  reference d6 base checkpoints).
- All oracle dumps run with `NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1` so the
  reference computes in the exact CPU fp32 path the port targets (the fused
  torch.compile optimizer kernels run eagerly).

The two exporter tools live in `../tools/`:
- `nanochat_export.py` — tokenizer + data + fixture exports.
- `nanochat_dump.py` — model/optimizer parity oracles.

## Neutral binary formats

Layouts are documented at their readers/writers in `tokenizer.zig` and
`data.zig`: `NCTOKz01` tokenizer.bin, `NCTKB_01` token_bytes.bin, `NCIDS_01`
encode fixtures, `NCTXT_01` trainer corpus, `NCDOC_01` framed docs, plus the
safetensors/JSON oracle dumps. All integers little-endian.

## Regenerate

Run from `refs/nanochat` with the venv python, `PYTHONPATH=refs/nanochat`,
`NANOCHAT_BASE_DIR=$HOME/.cache/nanochat`. Prefix all `--out` paths with
`refs/nanochat-goldens/`. (One-time setup: `python -m nanochat.dataset -n 8`,
`python -m scripts.tok_train --max-chars=2000000000`.)

Tokenizer + encode/train fixtures (`nanochat_export.py`):
- `export-tokenizer --out .` → `tokenizer.bin`, `token_bytes.bin`, `merges.txt`
- `dump-ids --out ids_parity.bin` → encode-parity fixture (`NCIDS_01`, 207 items)
- `export-train-text --out train_text_small.bin --max-chars 3000000` → trainer corpus
- `train-ref --input train_text_small.bin --vocab 1024 --out tokref_v1024.bin` → rustbpe merge-list gate

Data (`nanochat_export.py`):
- `export-docs --split train --out base_train_small.bin --max-docs 30000` (+ `.idx.json`)
- `export-docs --split val --out base_val.bin` (+ `.idx.json`)
- `export-sft-mixture --split val --out sft_mixture_val.jsonl --max-convs 2000`
- `dump-render --split val --out render_val.bin --n 64`
- `dump-base-batches --B 8 --T 512 --split val --out base_batches.bin --n-batches 16`
- `dump-sft-batches --B 4 --T 2048 --out sft_batches.bin --n-batches 4`

Model / optimizer oracles (`nanochat_dump.py`, for `--config d6` and `d2`):
- `dump-init --config d6 --out init_d6.safetensors` (+ `.config.json`, incl. `rms_norm_eps`)
- `make-batch --config d6 --B 2 --T 64 --out fixed_batch_d6.bin`
- `dump-forward --config d6 --init init_d6.safetensors --batch fixed_batch_d6.bin --out fwd_oracle_d6.safetensors` (+ a `--loss-reduction none` run → `fwd_oracle_d6_none.safetensors`)
- `dump-grad --config d6 --init … --batch … --out grad_oracle_d6.safetensors`
- `dump-optsteps --config d6 --init … --batch … --steps "1,10" --out-prefix optstep_d6` (+ `_schedule.json`)
- `dump-loss-trace --config d6 --init … --steps 50 --out loss_trace_d6.json --batches-out trace_batches_d6.bin`
- `dump-bpb --config d6 --init … --batches trace_batches_d6.bin --out bpb_oracle.json`
- `dump-greedy --config d6 --init … --prompt "The capital of France is" --max-tokens 32 --out greedy_stream_d6.json`
- `dump-sft-trace --out loss_trace_sft_d6.json` → SFT loss-trace + `trace_batches_sft_d6.bin` + `sft_schedule_d6.json`

Trained-model chat/SFT source (from a reference base checkpoint):
- Convert `~/.cache/nanochat/base_checkpoints/d6/model_00XXXX.pt` → `base_ckpt_d6_stepXXXX.safetensors` (strip `_orig_mod.` prefix, bf16→f32, `safetensors.torch.save_file`).
- `dump-greedy --init base_ckpt_d6_stepXXXX.safetensors …` → `greedy_trained_d6_stepXXXX.json` (the real-model greedy-parity target).
