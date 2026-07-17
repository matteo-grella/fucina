# ptqtp-qwen3 — PTQTP-decorate a Qwen3 GGUF and compare NLL

Qwen3 PTQTP decoration harness ([`main.zig`](main.zig)): load a
GGUF (any source dtype Fucina can decode — f16, Q4_K, Q8_0, … — the method
is source-agnostic), measure teacher-forced NLL/perplexity on a text file
**before** decoration, decorate every eligible layer linear with dual
trit-planes in place, measure NLL again on the same tokens, and finish with
a greedy completion plus decode timing. One process, one model load, direct
deltas. Method: [docs/PTQTP.md](../../docs/PTQTP.md); tuned per-size
recipes: [docs/PTQTP-RECIPE.md](../../docs/PTQTP-RECIPE.md).

## Getting the weights

Any Qwen3 dense GGUF works — see the weights table in
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md#getting-the-weights):

```sh
mkdir -p models
hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf --local-dir models
```

If your source only ships bf16 and you want f16, transcode locally:
`zig build export-gguf -Doptimize=ReleaseFast -- --from-gguf <src>.gguf --out models/Qwen3-0.6B-f16.gguf --dtype f16`.

## CLI

```sh
zig build ptqtp-qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-f16.gguf --nll docs/REFERENCE.md
zig build ptqtp-qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_M.gguf --nll docs/REFERENCE.md
zig build ptqtp-qwen3 -- models/Qwen3-0.6B-f16.gguf --planes 0   # undecorated baseline only
zig build ptqtp-qwen3 -Doptimize=ReleaseFast -- models/Qwen3-1.7B-BF16.gguf --planes 3 --save models/qwen3-1.7b-ptqtp-k3.gguf
```

The model path is positional; `--nll FILE` is any plain-text file (up to
16 MiB). Flags: `--planes 0|1|2|3` (default 2; 0 = no decoration),
`--nll-tokens N` (default 512 supervised tokens), per-projection overrides
`--down-planes N` / `--o-planes N`, `--head-planes N` (decorates the head
matmul only, independent of `--planes`; on tied models the embedding lookup
stays in source precision), `--skip-first N` / `--skip-last N` (leave edge
layers in source precision), `--save FILE`, `--prompt TEXT` (default
"The capital of Italy is"), `--max-new N` (default 48).

`--save` persists the decorated model as a GGUF (one standalone TQ2_0
tensor per trit-plane, everything else byte-verbatim) so the solve runs
once: the saved file loads through the ordinary qwen3 loaders — this
example, the chat CLI, speculation — with plane pair-detection, no
re-decoration. Loading a saved file here with `--planes 0 --nll FILE`
reproduces the decorated "nll after" exactly.

## Quick run

The Q8_0 file downloaded above works as-is (the method is source-agnostic),
and any plain-text file serves as the NLL text — a tracked repo file avoids
extra downloads:

```sh
zig build ptqtp-qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf --nll docs/REFERENCE.md
```

Output, in order: a `loaded …` line (layers, hidden size, vocab, load
seconds); `nll before: … (ppl …) over 512 supervised tokens`; one
decoration summary (`decorated N linears … in … s: …M weights -> … MiB
packed …, rms rel err …, unconverged groups …/…`); `nll after: …` on the
same tokens; the greedy completion for `--prompt`; and prefill ms +
decode tok/s. Decoration takes seconds at 0.6B, ~90 s at 1.7B K=3. With
`--planes 0` (and no `--head-planes`) only the `nll before` line prints —
that is the line that reproduces a saved file's decorated `nll after`.

Absolute NLL depends on the chosen text file; the before/after delta over
identical tokens is the signal. The recorded before/after perplexity
ladder per source dtype and plane count — including the from-Q4_K_M
graceful-degradation figures — is in [docs/PTQTP.md](../../docs/PTQTP.md)
§Measured.

## Shared knobs

The ReleaseFast/`-Dcpu` build discipline, GPU offload, and global
thread/BLAS knobs are shared machinery — see
[docs/RUNNING-MODELS.md](../../docs/RUNNING-MODELS.md). Embeddings,
lm_head, and norms stay in source precision unless `--head-planes` is
given.
