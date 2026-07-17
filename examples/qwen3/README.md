# Qwen3 — chat, generation, speculative decoding

`zig build qwen3` runs the Qwen3 family — dense 0.6B/1.7B/… and the 30B-A3B
MoE — from a GGUF file. It is the most complete runner: chat, REPL, raw
generation, speculative decoding, benchmarks, logit-parity tooling. Entry
file: [`main.zig`](main.zig); the installed binary is
`./zig-out/bin/fucina-qwen3`.

This README is the canonical reference for the sampling flags: the `gemma4`
and `diffusion-gemma` runners mirror the same set.

## Getting the weights

The repo does not ship any weights. All artifacts are GGUF files from
Hugging Face; the `hf` CLI (from `pip install -U huggingface_hub`) downloads
single files:

```sh
mkdir -p models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models
hf download unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf --local-dir models
```

- **Dense (0.6B/1.7B/…).** Official
  [`Qwen/Qwen3-0.6B-GGUF`](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) /
  [`Qwen/Qwen3-1.7B-GGUF`](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF) ship
  Q8_0 only; the full K-quant ladder (Q4_K_S … Q6_K + bf16) is on
  [`bartowski/Qwen_Qwen3-0.6B-GGUF`](https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF)
  or [`unsloth/Qwen3-0.6B-GGUF`](https://huggingface.co/unsloth/Qwen3-0.6B-GGUF)
  (same pattern per size).
- **MoE 30B-A3B.**
  [`unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF)
  — `…-Q5_K_M.gguf` (21.7 GB), `…-Q6_K.gguf` (25.1 GB).
- **Filenames.** bartowski prefixes files with `Qwen_` (e.g.
  `Qwen_Qwen3-0.6B-Q4_K_S.gguf`) — the runner takes a plain path, so rename
  or adjust as you like.

## Chat, REPL, completion

```sh
# Single-turn chat (streams the reply; --no-think skips the <think> phase)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "What is the capital of France?" --no-think

# Multi-turn interactive REPL (empty line or Ctrl-D quits)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --repl

# Chat with a system prompt + sampling overrides
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "Tell me a joke" --system "You are a pirate." \
  --temp 0.7 --top-k 40 --top-p 0.9 --seed 42

# The big MoE (20 GB — give it a moment to mmap)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf \
  --chat "Explain quantum entanglement in one paragraph." --no-think

# Raw completion from a text prompt (greedy unless sampling flags given)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "The capital of France is" --gen 64
```

Chat needs the GGUF's tokenizer + chat-template metadata (both present in
the artifacts above).

## Speculative decoding

```sh
# Lossless speculative decoding (SAM + Token-Recycling cascade; prints acceptance stats)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  --prompt "..." --gen 128 --spec

# Speculative decoding with an injected reference document (the RAG seam:
# the drafter can copy spans from the injected text)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  --prompt "Summarize the doc" --gen 128 --spec --spec-ref doc.txt
```

`--spec` also works in `--chat`/`--repl`. `--spec-ref` repeats (up to 8
files). `--spec-bench` measures the verify economics instead of generating:
one batched k-token verify forward vs k single steps, for k in
{2, 4, 8, 16}, best of max(`--bench`, 5) reps.

## Benchmarks

```sh
# Warm prefill/decode benchmark, fair vs llama-bench (load once, best-of-R)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf \
  <prompt-token-ids> --gen 64 --bench 5

# Batched multi-stream decode: N lockstep streams (one m=N weight pass/step)
# vs N sequential runs, aggregate tok/s + token-for-token cross-check
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  <prompt-token-ids> --gen 64 --bench 3 --streams 4
```

`<prompt-token-ids>` is a positional comma-separated id list (default
`151644,872,198,9707`); `--prompt` text works here too.

The measurement protocol (matched `llama-bench` invocation, the
prompt-length matrix, thermal discipline) is in
[`docs/BENCHMARK.md`](../../docs/BENCHMARK.md).

## Parity oracles

```sh
# Tokenizer-parity oracle (one token id per line; no weights loaded)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --tokenize input.txt

# Logit parity vs another implementation (raw little-endian f32 dump/compare)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  151644,872,198,9707 --logits-out /tmp/f.bin
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  151644,872,198,9707 --compare-logits /tmp/ref.bin
```

Without `--gen`, the runner performs a single forward pass over the token
ids and prints load/forward timing plus the top-5 logits — that is the path
`--repeat`, `--profile`, `--logits-out` and `--compare-logits` serve.
`--verify-cache N` cross-checks cached vs full attention over N steps.

The reference side of the logit compare comes from the pinned llama.cpp
checkout: `tools/fetch_refs.sh llama.cpp --build` clones it under `refs/`
(gitignored) and builds the CPU-only binaries into
`refs/llama.cpp/build-cpu/bin/` (`llama-debug` is an extra target:
`cmake --build refs/llama.cpp/build-cpu --target llama-debug`). The
dump/compare recipe — `llama-debug --save-logits` on the same token ids,
then `--compare-logits` on the dump, with the expected-drift guidance for
quantized formats — is in [`docs/BENCHMARK.md`](../../docs/BENCHMARK.md)
under "Correctness check"; `tools/llama_logits.cpp` is the standalone
last-token dumper compiled against a llama.cpp checkout (usage:
`<model.gguf> <comma-ids> <out.bin>`). For `--tokenize`, the comparison
target is `llama-tokenize --ids --no-escape` from the same build
([`docs/SPECULATIVE.md`](../../docs/SPECULATIVE.md)).

## KV cache

```sh
# q8_0 KV cache (halves KV memory — capacity option; decode is NOT faster on M1)
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --prompt "..." --gen 256 --cache-type q8_0
```

`--kv-save[=PATH]` adds crash-safe KV persistence for `--chat`/`--repl`:
conversations reopen warm across process restarts (default sidecar
`<gguf>.kvcache`).

## Flags

Every value flag takes both `--flag value` and `--flag=value`, except the
MoE streaming knobs (`--moe-cache-mb`, `--moe-cache-slots`, `--moe-pin-mb`,
`--moe-expert-top-p`), which are `=`-form only.

Modes and generation:

| flag | meaning |
| --- | --- |
| `<model.gguf>` | model file (first argument, required) |
| `<token-ids>` | positional comma-separated prompt token ids (default `151644,872,198,9707`) |
| `--chat MSG` | single-turn chat, streams the reply |
| `--repl` | multi-turn interactive REPL (empty line or Ctrl-D quits) |
| `--system MSG` | system prompt for `--chat`/`--repl` |
| `--no-think` | skip the `<think>` phase (also switches the chat sampling defaults) |
| `--prompt TEXT` | raw completion: encode TEXT as the token stream |
| `--gen N` | generate N tokens (selects the generation paths) |
| `--stop TOKEN_ID` | stop generation at a token id |
| `--info` | print model config/tokenizer info and exit |
| `--tokenize FILE` | encode a text file, one token id per line (no weights loaded) |

Sampling (the canonical set — `gemma4` and `diffusion-gemma` mirror it):

| flag | meaning |
| --- | --- |
| `--temp F` | temperature (0 = greedy argmax) |
| `--top-k N` | top-k cutoff (0 = off) |
| `--top-p F` | nucleus cutoff (1.0 = off) |
| `--min-p F` | min-p cutoff (0 = off) |
| `--repeat-penalty F` | repetition penalty (1.0 = off) |
| `--seed N` | RNG seed |

Defaults: chat samples with Qwen3's recommended settings — temp 0.6,
top-k 20, top-p 0.95, min-p 0, repeat-penalty 1.0, seed 0; with
`--no-think`, temp 0.7 and top-p 0.8. The completion and benchmark paths
default to greedy (temp 0, top-k 0, top-p 1.0). Flags override either set.

Speculative decoding and benchmarks:

| flag | meaning |
| --- | --- |
| `--spec` | lossless speculative decoding (SAM + Token-Recycling cascade; prints acceptance stats) |
| `--spec-ref FILE` | inject a reference document the drafter can copy spans from (repeatable, up to 8) |
| `--spec-bench` | verify-economics microbenchmark (batch-k verify vs k single steps, k in {2,4,8,16}) |
| `--bench R` | warm prefill/decode benchmark, best-of-R (load once) |
| `--streams N` | with `--gen`: N lockstep decode streams vs N sequential runs |
| `--repeat N` | re-run the plain forward N times |
| `--profile` | per-block timings |

Parity and KV cache:

| flag | meaning |
| --- | --- |
| `--logits-out PATH` | dump last-token logits, raw little-endian f32 |
| `--compare-logits PATH` | compare last-token logits against such a dump |
| `--verify-cache N` | cached-vs-full attention check over N steps |
| `--cache-type f16\|q8_0` | KV cache dtype (default `f16`) |
| `--kv-save[=PATH]` | crash-safe KV persistence for `--chat`/`--repl` (default `<gguf>.kvcache`) |

Constrained decoding (needs a `-Dllguidance=true` build):

| flag | meaning |
| --- | --- |
| `--json-schema J\|@F` | reply must satisfy a JSON schema |
| `--lark G\|@F` | reply must satisfy a Lark grammar |
| `--regex P` | reply must satisfy a regex |

The three grammar flags are mutually exclusive; `@F` reads the grammar from
a file. Usage guidance (composition with `--no-think`/`--spec`/`--streams`,
sampling advice, examples) is in
[`docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md) and REFERENCE.md
§13.6.

Interactions: `--streams` ignores `--spec` and `--stop` (all streams run the
full length); `--spec` with `--bench` R>1 is ignored (`--bench` is the
plain-decode protocol); `--spec-bench` excludes the grammar flags.

## Shared knobs

Build discipline (`-Doptimize=ReleaseFast`, `-Dtarget`/`-Dcpu`), global
thread/BLAS knobs, GPU offload (`-Dgpu=metal`/`-Dgpu=cuda`), constrained
decoding usage, and the MoE expert-streaming machinery are documented once
in [`docs/RUNNING-MODELS.md`](../../docs/RUNNING-MODELS.md). This runner
accepts the full `--moe-stream` knob set (`--moe-cache-mb`,
`--moe-cache-slots`, `--moe-pin-mb`, `--moe-no-learn`, `--moe-pilot`,
`--moe-expert-top-p`) for out-of-core MoE models bigger than RAM — see
"Streaming MoE experts from disk" there.
