# Contributing to Fucina

Contributions are welcome — kernels, model families, ports, tooling, docs,
benchmark records. This file is short because the bar is simple: a human who
understands the change, and evidence on the two tracks the project actually
regresses on.

## A human sends the PR

We expect you to work with coding agents — this project is built that way,
and the README says so. Agents sometimes produce kernels or algorithms more
sophisticated than their operator would write by hand; that is fine, and it
is not grounds for rejection. What we require is a human who stands behind
the change: you understand what it does and how it fits the architecture,
you ran the gates and it regresses neither track, and you are confident
enough in its quality to answer for it in review. Line-by-line authorship
is not the bar — accountability is. PRs with no such human behind them
(review questions unanswered, or bounced back as unfiltered model output)
are closed without ceremony.

## The two regression tracks

Fucina regresses in exactly two ways: it becomes **wrong**, or it becomes
**slow**. Test your change against the failure modes it can realistically
affect — no more, no less. A doc fix needs no benchmark; a tokenizer change
needs the parity oracles but not a GEMM sweep; a kernel change needs both
tracks, always.

**Correctness.** `zig build test` must be green, and run the variants your
change can affect: `-Dbackend=scalar` and `-Dblas=none` for anything
numeric (the scalar backend is the reference — native and scalar must
agree), the parity oracles for anything touching a model family (logit
parity, token-ID-exact tokenization, byte-exact quant encoding — see the
family's example runner). `zig build arch-check` and `zig build doc-check`
guard structure and docs.

**Speed.** Perf claims go through the paired gate — protocol in
`docs/BENCHMARK.md`, one command in practice:

```sh
tools/fetch_refs.sh --build
python3 tools/bench_gate.py --models qwen3-0.6b-q6_k --tasks prefill,decode
```

Respect the thermal discipline documented there; on laptops a hot chip
inverts results. A kernel/perf change is not done until it is measured.

**Report what you did.** In the PR (or commit notes), include: the exact
commands you ran, the machine and backend configuration (CPU, OS, threads,
`-Dblas`/`-Dbackend` flags), the model and quantization used, and any
notable failures or skipped suites. "Tests pass" without the machine and
the commands is not a report.

## Backend and hot-path changes

Anything under `src/backend/`, `src/exec/`, or a model family's forward
path affects inference for every runner built on it. Do not send such a PR
without checking **both** tracks: correct-but-slower and fast-but-wrong are
both regressions, and neither is accepted on its own. The single exception:
a speed penalty is acceptable when it is the unavoidable cost of fixing a
real correctness bug — say so explicitly in the PR, with the before/after
numbers, so the trade is a recorded decision rather than a surprise.

## Provenance

If your change ports or adapts third-party code, say where it came from.
The license must be compatible with MIT, the source gets credited in
`docs/THIRD-PARTY-NOTICES.md`, and faithful ports should name the upstream
file they follow (this repo's convention — see the tokenizers and quant
encoders). Uncredited ports are treated as bugs.

## House rules

`AGENTS.md` carries the working conventions (ownership and deinit
discipline, exhaustive switches, tests in sibling `_tests.zig` files,
surgical changes). Match the style of the code you touch. By contributing
you agree your work is released under the repository's MIT license.
