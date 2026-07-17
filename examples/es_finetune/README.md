# es-finetune — evolution-strategies fine-tuning (gradient-free)

`es-finetune` is [`finetune`](../finetune/README.md)'s ES-at-scale twin
(arXiv 2509.24372): same model loading, same built-in pirate dataset and
`--data` JSONL path, same deterministic `--shuffle` loader, same checkpoint
directory layout, and the same trainer forward for evaluation. What changes
is the learning signal: no backward pass, no optimizer state — `fucina.es`
perturbs the parameters with seed-regenerated gaussian noise, scores each
population member with a reward, and applies the ES-at-scale update
(z-scored rewards, alpha/population scaling, no 1/sigma).

Entry point: [`main.zig`](main.zig)
(`zig build es-finetune`). Like `finetune`, it prints a BEFORE vs AFTER
greedy continuation for one held prompt around the run, plus per-iteration
reward statistics.

## Parameter sets (`--mode`)

- `lora` (default): perturbs only the LoRA q/v adapters — `finetune`'s
  target set; B starts at zero exactly like the backprop run. Cheap
  perturb/update, and the model's frozen weights may stay quantized.
- `full`: perturbs every resident float weight of the base model
  (embeddings, projections, norms) — the paper's full-parameter setting.
  Requires an f32/f16/bf16 GGUF: quantized blocks cannot take gaussian
  noise.

## Rewards (`--reward`)

- `rule` (default): DeepSeek-R1-style rule reward on greedy generations —
  `unigram-F1(generated, gold) + 0.1 * format`, where `format` checks the
  response envelope (`--format-prefix`/`--format-suffix`; defaults match the
  pirate dataset's "Ahoy! … matey."). Generation-bound: population × batch
  greedy continuations per iteration.
- `acc`: bounded teacher-forced composite `token_accuracy + 0.1 * exp(-CE)`
  — one forward per sample, dense, softly self-stopping at saturation.
  Prefer it for loss-style training.
- `nll`: raw negative mean cross-entropy of the gold response — directly
  comparable with `finetune`'s loss curve, but unbounded: on long runs one
  catastrophic member dominates the z-score and the run degrades past its
  peak (pair with `--norm centered_ranks` and `--save-every` selection).

Every population member scores the same per-iteration sample batch
(`--batch`); member evaluation runs in place (perturb → score → restore), so
one model instance serves the whole population.

## Commands

```sh
# ES fine-tune the LoRA q/v adapters (quantized base is fine — forward passes only).
# --reward acc = bounded token-accuracy + 0.1*exp(-CE) composite (recommended: cheap AND stable);
# --reward nll = raw negative CE (loss-comparable with finetune, but unbounded — degrades on
#                long runs unless paired with --norm centered_ranks and --save-every selection);
# --reward rule = R1-style rule reward on greedy generations (generation-bound, slower).
zig build es-finetune -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-Q4_K_S.gguf \
  --reward acc --iterations 100 --population 8 --batch 5 --sigma 0.02 --save-every 25 \
  --save /tmp/qwen3-es

# Full-parameter ES (the paper's setting): every resident float weight of the model.
# Needs an f32/f16/bf16 GGUF — quantized blocks cannot take gaussian noise.
zig build es-finetune -Doptimize=ReleaseFast -- --model models/Qwen3-0.6B-f16.gguf \
  --mode full --reward nll --iterations 10 --population 4 --batch 2 --sigma 0.001

# Useful extras: --alpha F (default sigma/2)  --noise iid|correlated  --antithetic  --norm z_score|centered_ranks
#                --restore regenerate|snapshot  --anchor-decay l1|l2 --anchor-lambda F (AWD anti-drift)
#                --max-new N + --format-prefix/--format-suffix (rule reward)  --load DIR (resume:
#                (seed, es_iteration) regenerate the population stream)  --data/--shuffle/--data-seed
```

## Flags

| flag | default | meaning |
| --- | --- | --- |
| `--model PATH` | `models/Qwen3-0.6B-Q4_K_S.gguf` | base GGUF (`--mode full` needs f32/f16/bf16) |
| `--mode lora\|full` | `lora` | parameter set (see above) |
| `--reward rule\|acc\|nll` | `rule` | reward (see above) |
| `--iterations N` | 20 | ES iterations |
| `--population N` | 8 | population members per iteration |
| `--sigma F` | 0.02 | perturbation scale |
| `--alpha F` | sigma/2 | ES learning rate |
| `--noise iid\|correlated` | `iid` | noise scheme |
| `--antithetic` | off | antithetic sampling (mirrored member pairs) |
| `--norm z_score\|centered_ranks` | `z_score` | reward normalization |
| `--restore regenerate\|snapshot` | `regenerate` | how a member's perturbation is undone |
| `--anchor-decay l1\|l2` + `--anchor-lambda F` | off | anchored weight decay (AWD anti-drift) |
| `--cache-streams` | off | cache each iteration's noise streams in RAM (bitwise identical to regeneration; benchmark before enabling) |
| `--batch N` | 3 | samples scored per iteration (shared by all members) |
| `--max-new N` | 20 | generation cap (rule reward) |
| `--format-prefix S` / `--format-suffix S` | `"Ahoy!"` / `"matey."` | rule-reward format envelope |
| `--rank N` | 8 | LoRA rank (lora mode) |
| `--lora-alpha F` | 16 | LoRA scaling alpha (lora mode; `--alpha` is the ES learning rate here) |
| `--seq-max N` | 256 | token cap per encoded training pair |
| `--data PATH.jsonl` | built-in set | JSONL SFT dataset (`src/llm/data.zig`) |
| `--shuffle` / `--data-seed N` | off / `--seed` | deterministic per-epoch shuffle |
| `--save DIR` | `/tmp/fucina-qwen3-es` | checkpoint directory |
| `--save-every N` | 0 (final save only) | periodic checkpoint interval |
| `--load DIR` | — | resume: (seed, es_iteration) regenerate the population stream |
| `--seed N` | 42 | RNG seed |

## Checkpoints and the serve loop

Checkpoints hold `adapters.safetensors` (lora) or `model.safetensors` (full)
plus `trainer_state.json` with the ES fields — there is no
`optimizer.fucina`, because ES has no optimizer state; `(seed, es_iteration)`
fully regenerate the population stream on `--load`.

The lora checkpoint is the same `adapters.safetensors` `finetune` writes, so
the [merge → quantize → serve loop](../finetune/README.md) applies unchanged.

## Getting the weights

Same sources as `finetune`:
[Getting the weights](../../docs/RUNNING-MODELS.md#getting-the-weights) in
docs/RUNNING-MODELS.md. `--mode lora` runs on any Qwen3 dense GGUF
(quantized included); `--mode full` needs a float GGUF such as
`Qwen3-0.6B-f16.gguf` — if your source only ships bf16, transcode one
locally with the f16 note in that section.

## Further reading

- [docs/TRAINING.md](../../docs/TRAINING.md) §13 — evolution strategies: the
  stability analysis behind the reward/norm choices and AWD.

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, `-Dcpu`), global thread/BLAS
knobs, GPU offload, and the other shared machinery are documented centrally
in [docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). Serving an
ES-tuned model goes through the same `qwen3`/`lmserve` steps as the
[finetune loop](../finetune/README.md).
