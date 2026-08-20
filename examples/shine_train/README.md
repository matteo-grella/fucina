# shine-train — train the SHINE hypernetwork (LoRA or cartridge readout)

Zig-native training for the SHINE generator (REFERENCE.md §13.12) over a
frozen dense qwen3 base: the Meta-LoRA, the M2P metanetwork, and the
positional embeddings learn to compile a context into generated
parameters. `--rows N` trains the **cartridge readout** (context -> a
standard KV-prefix cartridge with N rows per layer); `--lora-r N` trains
the paper's **LoRA readout**. Exactly one of the two.

Entry point: [`main.zig`](main.zig) (`zig build shine-train`).

## Data

Triple JSONL, one object per line:

```json
{"evidence": "the document text ...", "instruction": "a question about it", "response": "the answer"}
```

Evidence is tokenized raw (no template, the reference's collator);
instruction/response use the chat template with prompt-masked labels (the
finetune example's encoding). `tools/export_squad_triples.py` produces
train/eval splits from the SQuAD parquet, holding out whole CONTEXTS so
the eval loss measures unseen documents.

## Train

```sh
# Cartridge readout at 0.6B: 88 rows/layer -> M = 176 memory tokens
# (the same generator budget as the LoRA-readout smoke checkpoints)
zig build shine-train -Doptimize=ReleaseFast -Dgpu=metal -- \
  --model models/Qwen3-0.6B-BF16.gguf --rows 88 \
  --data squad_train.jsonl --eval-data squad_eval.jsonl \
  --steps 2000 --lr 1e-4 --save shine-cartridge-0.6b.safetensors

# The LoRA readout trains with the same loop: swap --rows for --lora-r 8
```

`--save` writes the trainable leaves as a safetensors state dict;
`--load` resumes it (strict name match). The held-out eval line is the
smoke-run instrument: the mean loss over UNSEEN contexts must fall
clearly below its step-0 value for the readout to be doing its job.

## Flags

| flag | default | meaning |
| --- | --- | --- |
| `--model PATH` | required | frozen qwen3 base GGUF |
| `--data PATH.jsonl` | required | training triples |
| `--rows N` \| `--lora-r N` | pick one | readout and budget (M from the identity) |
| `--eval-data PATH` | — | held-out triples (unseen contexts) |
| `--steps N` | 200 | optimizer steps (one triple per step) |
| `--lr F` | 1e-4 | AdamW learning rate |
| `--metalora-r N` | 128 | Meta-LoRA rank (encoder side paths) |
| `--scale F` | 0.001 | readout scale (`sqrt(scale)` on the sliced output) |
| `--evidence-max N` | 512 | context token cap |
| `--seq-max N` | 1024 | conversation token cap |
| `--eval-every N` | 50 | held-out eval cadence |
| `--save F` / `--save-every N` / `--load F` | — | leaf state dict checkpointing |
| `--seed N` | 42 | leaf init seed (bitwise deterministic) |

## Test it like a normal cartridge

A trained cartridge-mode run is evaluated with the SAME instruments as
distilled cartridges: generate per-context artifacts and serve them
through the standard stack. The trained leaves feed the qwen3 runner
once exported as a shine GGUF (writer pending); until then, in-driver
eval loss over held-out contexts plus `--shine-save-cartridge` from a
converted checkpoint cover the loop. Baselines for the comparison
matrix: no-context, ICL (context in prompt), and a DISTILLED cartridge
(`zig build cartridge`) at the same `p` on the same held-out documents.
