# Benchmarks — Fucina vs the reference implementations, on CPU

This file is the benchmark record for Fucina's runners: the measurement
protocol, and results against each family's reference implementation on the
same machine, same weights, same thread count, CPU-only on both sides —
llama.cpp for the LLM runners (the bulk of this file), parakeet.cpp for
ASR, omnivoice.cpp for TTS. The NAM example is parity-oriented
(NeuralAmpModelerCore); its record lives in `examples/nam/README.md`.
**The record is one snapshot, taken as of 2026-07-04** (llama.cpp build
30af6e2 throughout; every reference is pinned to its exact commit in
`tools/fetch_refs.sh`).

Two ground rules for reading it:

- **Every number carries its hardware and measurement conditions.** CPU
  benchmarks on laptops are thermally and page-cache sensitive and
  shape-specific; a number without its conditions is not a result.
- **Losses are recorded as plainly as wins.** Where llama.cpp is faster, this
  file says so, with the measured ratio.

## Scoreboard

Condensed from the records below. "Ratio" is always Fucina / llama.cpp
throughput (>1 = Fucina faster).

**Apple M1 Max (arm64, 8 threads, macOS — llama.cpp runs with its Accelerate
BLAS backend, its default on this platform):**

Fucina wins:

- **Dense prefill across the board** (Qwen3 0.6B all seven formats + 1.7B,
  20 prompt lengths each): per-format geomeans 1.18–1.81x. llama.cpp
  switches to its Accelerate/AMX path at batch >= 32, and that transition
  costs it heavily at pp32–129 (up to 7.2x in Fucina's favor at f16 pp32);
  1.7B pp256 is 1.07x under the prewarmed paired gate.
- **Large-batch prefill on the 0.6B** (pp256): 1.26–1.42x across the
  quantized formats and 2.46x on f16.
- **Large MoE prefill** (pp64–256): Qwen3-30B-A3B 1.44–2.09x;
  Gemma-4-26B-A4B 1.66–1.84x measured cool.
- **Gemma-4-26B mid-batch prefill** (pp4–33): ahead on all 11 measured
  lengths same-session cool, 1.17–2.15x (the pp4–9 cells await one
  pristine-machine confirmation — see that section's disclosure).
- **Dense decode**: Q4_K_M/Q4_K_S +15%, Q8_0 +14%; f16/Q5_K/Q6_K parity or
  better cool (Q6_K paired gate 1.06x).
- **30B MoE decode and large prefill** (Q5_K_M, prewarmed page cache both
  sides): decode tg32 1.10–1.36x, pp256 1.43–1.87x.
- **30B MoE mid-batch prefill** (pp15–33, after the 2026-07-10 scheduler
  fix): M1 self-A/B +27–37% (68→87 tok/s at pp15, 76→105 at pp32,
  interleaved same-day); the x86 band flipped from 0.36–0.52x behind to
  0.97–1.22x (decomposition in the x86 section).
- **Lossless speculative decoding** (no draft model): up to 2.3x on
  retrieval-structured tasks, with a cost gate whose worst measured case is
  0.98–0.99x (see `SPECULATIVE.md`).
- **Batch-N multi-stream decode**: 3.19–3.25x aggregate throughput at N=8
  streams vs running the streams sequentially.

llama.cpp wins:

- **Qwen3.5-0.8B Q8_0 pp32: 0.86x** (3 interleaved rounds, both orders).
  The loss is confined to this shape — pp128 1.09x, pp512 1.17x, and decode
  1.37x all win. Fucina's pp32 samples are bimodal across processes
  (408–683 tok/s, best samples at llama parity or above) while llama's are
  tight; profiling shows the slow processes uniformly inflated across every
  parallel phase. Open item (see Recorded negatives for what has been ruled
  out).
- **30B Q6_K decode: 0.88x** — recorded without page-cache prewarming; its
  Q5_K_M sibling measures 1.36x under the stricter prewarmed protocol, so
  this cell is likely conditions-bound, but it stands until re-measured.
- **30B/26B MoE prefill at pp256 with Q4_K-transcoded experts**: Fucina
  ~15–17% behind in the same-session comparison (the transcoded GGUFs were
  not kept, so this row awaits a rebuild to re-measure).

**Intel i9-13950HX (x86-64, Raptor Lake 8P+16E, AVX2 + AVX-VNNI, no AVX-512,
Linux, no BLAS on either side — verified, see below):**

Fucina wins:

- **Dense Qwen3-0.6B, all quantized formats**: paired-gate median ratios
  1.32–1.95 per format, maxima up to 2.56 (Q5_K_S pp128). Q8_0 pp256 1.45x,
  Q4_K decode 1.21–1.25x.
- **f16 model**: no llama.cpp pairing recorded, but the f32-accumulate f16
  GEMM took pp1024 from 17.9 to 354 tok/s on this box (Fucina-only A/B).

llama.cpp wins:

- **Qwen3-30B-A3B MoE mid-batch prefill — resolved 2026-07-10.** The
  pp15–33 band went from 0.36–0.52x to Q5_K_M 1.15–1.22x and Q6_K
  0.97–1.05x (same-day paired reruns; a reverted-gate control first
  reproduced 0.375–0.509, so the llama side is directly comparable). The
  loss was scheduling, not kernels: the band ran monolithic per-expert
  tasks below the old 512-pair phased-chain gate, so the chain machinery
  never executed there. Gating the chain at 64 pairs recovers most of the
  band; small-m column chunking lifts Q6_K over parity at pp15–17.
  Residual: Q6_K pp31–33 at 0.965–0.987x (open, small); pp1–7 stays on the
  monolithic path below the 64-pair gate.
- **MoE decode**: 0.90–0.95x (weight-bandwidth-bound at m=1).
- **Q5_K dense decode**: 0.90x (the m=1 GEMV path; open item). Q6_K dense
  decode is 0.987 — coin-flip parity.
- **Gemma-4-26B small-batch prefill** (pp1–9): 0.85–0.99x; Fucina wins pp15+
  (up to 1.31x) and decode is parity (0.996–1.000).

Results age: llama.cpp advances continuously. Treat everything here as
"measured as of the snapshot date, on that machine".

## Methodology

### Principles

- Build Fucina on the benchmarking machine itself, with no `-Dtarget`: the
  default native target compiles in the host's full ISA features. A
  cross-built baseline binary benchmarks the wrong kernels.
- Run one benchmark process at a time. Never run Fucina and llama.cpp in
  parallel.
- CPU-only llama.cpp runs: `-ngl 0`.
- Same model file (same quantization) on both sides.
- Treat prompt length as a benchmark parameter. Never quote `pp4` alone.
- Report both latency and tokens/sec. For Fucina, tokens/sec is
  `prompt_tokens * 1000 / avg_ms`.
- Keep correctness separate from throughput. Throughput may use synthetic
  token IDs; correctness compares logits for a fixed token sequence and
  requires top-token alignment. `tools/llama_logits.cpp` is the reference-side
  half of that check: compile it against a local llama.cpp checkout to dump
  per-position logits for a token sequence, then compare against the Fucina
  runners' `--logits-out`.
- Re-run after thermal or scheduler-sensitive changes. Back-to-back runs on
  laptops move by several percent — or much more (see thermal discipline).

### Why prompt length matters

CPU kernels have tile sizes (e.g. 4-row packing for SIMD lanes); models do
not. A correct benchmark matrix includes arbitrary lengths around SIMD/tile
and parallelism thresholds, not only powers of two: tail-only paths
(`pp1..pp3`), one exact tile (`pp4`), tile-plus-tail (`pp5`), near-tile tails
(`pp6, 7, 9, 15, 17`), boundary checks (`pp31, 33, 64, 127, 129`), and
multi-tile lengths (`pp8, 16, 32, 128, 256`). The routine matrix:

```text
1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,64,127,128,129,256
```

### The paired benchmark gate

`tools/bench_gate.py` is the tool behind any "parity-or-faster" claim. It is
deliberately conservative:

- every row runs in both process orders (`Fucina -> llama`, then
  `llama -> Fucina`), so order effects and thermal drift show up in the
  samples;
- rows whose cross-sample coefficient of variation exceeds `--max-cv`
  (default 8%) are reported **NOISY**, not counted as results — cool down
  and rerun;
- raw stdout/stderr and exact command lines are saved for every subprocess
  (under `compare/bench-gate-<timestamp>/`: `SUMMARY.md`, `results.json`,
  `results.tsv`, `raw/`);
- it compares **median** tok/s and exits nonzero when Fucina is below
  `--min-ratio` (default 1.0).

```sh
zig build -Doptimize=ReleaseFast
python3 tools/bench_gate.py \
  --models qwen3-0.6b-q6_k \
  --tasks prefill,decode \
  --lengths 1,2,3,4,5,6,7,8,9,15,16,17,31,32,33,64,127,128,129,256 \
  --rounds 1 --fucina-reps 3 --llama-reps 3 --cooldown-s 30
# or, after build args:
zig build bench-gate -- --models qwen3-0.6b-q6_k --tasks prefill,decode
```

`--list-models` prints the model catalog; `--llama-bench` points at your
llama-bench binary (default `refs/llama.cpp/build-cpu/bin/llama-bench`);
`--prewarm-model` reads the GGUF before each subprocess to stabilize the
page cache (important for 20+ GB models).

**Standing fairness note (cuts in Fucina's favor):** Fucina's prefill/decode
benchmark paths still perform the final logits/sampler work that
`llama-bench` skips inside its pp/tg loops. Passing rows are therefore
conservative for Fucina; failing rows deserve a no-logits A/B before being
called true kernel regressions.

### The op-level regression gate

`tools/opbench_gate.py` covers what the paired gate cannot: dispatch
latency, autograd tracking overhead, and training forward/backward
throughput have no llama.cpp counterpart, so they gate against a locally
recorded per-machine baseline instead of an external reference.

It drives the `bench*` build steps (facade, mlp, backward-diamond,
attention-backward, einsum, ce, scatter, optim) in ReleaseFast with the
production allocator and applies three rules:

- timings gate on the **median of N repeats** within `--tol` (default 10%);
  rows whose cross-repeat coefficient of variation exceeds 12% report
  **NOISY** instead of failing, as in the paired gate, and every timing
  exceedance is confirmed by re-running its suite after a cooldown
  (`--retry-cooldown-s`, default 30 s) — transient background load or a
  heat-soaked SoC must not fail the gate, and a real kernel regression
  reproduces;
- **`allocs_per_op` gates exactly** — allocation counts are deterministic,
  so any increase fails and any decrease asks for a re-record;
- **checksums gate exactly** — a checksum change is numerical drift, never
  noise.

```sh
python3 tools/opbench_gate.py record          # once per machine/toolchain
python3 tools/opbench_gate.py check           # nonzero exit on regression
python3 tools/opbench_gate.py check --suites facade,mlp --repeats 5
```

Baselines land in `bench/baselines/opbench-<host>.json` keyed to hostname,
arch, and Zig version; `check` refuses a mismatched environment without
`--force`. `zig build bench-check` is the compile-only companion: it builds
every bench executable without running one, so bench mains can no longer
rot unnoticed (five of them did exactly that before this step existed).

### Manual commands

Fucina (comma-separated token IDs; the values are fixtures, the length is
the variable):

```sh
zig build -Doptimize=ReleaseFast qwen3 -- models/Qwen3-0.6B-Q6_K.gguf TOKEN_IDS --repeat 100
zig build -Doptimize=ReleaseFast qwen3 -- models/Qwen3-0.6B-Q6_K.gguf TOKEN_IDS --repeat 500 --profile
```

llama.cpp, matching lengths:

```sh
refs/llama.cpp/build-cpu/bin/llama-bench -m models/Qwen3-0.6B-Q6_K.gguf -ngl 0 -t 8 -p N -n 0 -r 20 -o md
```

### Correctness check

Compare final logits against llama.cpp's debug logits for an explicit
token-id prompt:

```sh
refs/llama.cpp/build-cpu/bin/llama-debug \
  -m models/Qwen3-0.6B-Q6_K.gguf \
  -p ids:9707,847,829,374 \
  -ngl 0 -t 8 -tb 8 -b 4 -ub 4 \
  --save-logits --logits-output-dir compare/llama-q6-f32kv

zig build -Doptimize=ReleaseFast qwen3 -- \
  models/Qwen3-0.6B-Q6_K.gguf \
  9707,847,829,374 \
  --compare-logits compare/llama-q6-f32kv/llamacpp-Qwen3-0.6B-Q6_K.bin
```

Expected signal for quantized formats: top-token alignment plus bounded
logit drift. Exact bit equality is not expected.

### Hardware and reference builds

- **Apple M1 Max**, macOS, 8 threads both sides (`-t 8` for llama-bench;
  Fucina's build default is 8 workers). Fucina built
  `-Doptimize=ReleaseFast`. The llama.cpp reference for the M1 records is
  **build 30af6e2**, CPU-only at run time (`-ngl 0`),
  with its Accelerate BLAS backend active — that is llama.cpp's production
  configuration on macOS and it is kept deliberately (Accelerate drives
  Apple's AMX units; beating it is part of the job).
- **Intel i9-13950HX** (Raptor Lake, 8 P-cores + 16 E-cores, AVX2 +
  AVX-VNNI, no AVX-512), Linux, ReleaseFast. The llama.cpp comparison build
  was verified — checked, not assumed — as `GGML_BLAS=OFF` (no BLAS
  linkage), `GGML_NATIVE=ON` (`-march=native`, its AVX-VNNI kernels live),
  `GGML_LLAMAFILE=ON` (vendored tinyBLAS sgemm), `GGML_CPU_REPACK=ON`,
  `GGML_OPENMP=ON`. Both engines run their own CPU kernels with no BLAS on
  either side. The llama-bench binary did not embed a git commit
  (`build_commit: unknown` in its JSON output); the build flags above are
  the recorded provenance. llama-bench ran `-t 8` unpinned (the scheduler
  places threads on P-cores); Fucina ran its default 8 workers. Single-core
  rows used `taskset` to pin one P-core on both sides.

### Thermal discipline (Apple Silicon)

The single largest source of wrong conclusions in this file's history was
chip temperature:

- Heat soak **inverts thread scaling** (a throttled chip favors fewer active
  cores) and inflates per-phase profile wall times. Two documented dead-end
  investigations came from hot-chip profiles.
- Long sweeps (~40 min) systematically depress the rows measured late.
  Several apparent "llama.cpp wins decode" readings evaporated when
  re-measured cool and prewarmed — the discipline distinguishes real gaps
  from artifacts, and the record keeps only what survives it.
- Authoritative comparisons are **cool, isolated, best-of A/B pairs** with
  pre-cooldowns (30–240 s), ideally interleaved (ABBA) so drift hits both
  sides.
- For 20+ GB models, prewarm the page cache before decode A/Bs — a fresh
  process faults the weights cold and depresses whichever side runs first.
- Confirm the binary is ReleaseFast and the machine is not swapping. One
  recorded bad run traced to an accidentally-Debug binary plus swap pressure
  from concurrent model loads.

### Model files

The GGUF weights are **not** in this repository. Download them (e.g. from
Hugging Face) and place them under `models/` using the file names in the
`tools/bench_gate.py` catalog (`--list-models`), e.g.
`models/Qwen3-0.6B-Q6_K.gguf`, `models/Qwen3-30B-A3B-Instruct-2507-Q5_K_M.gguf`.
The reference implementations live under `refs/` — untracked local clones,
never vendored. `tools/fetch_refs.sh` clones every reference at the exact
commit this snapshot was measured against, and `tools/fetch_refs.sh
--build` additionally builds llama.cpp CPU-only into
`refs/llama.cpp/build-cpu/` (Accelerate stays on under macOS — llama.cpp's
production configuration there, kept deliberately; plain native CPU
elsewhere). `bench_gate.py --llama-bench` overrides the binary path.
`tools/fetch_refs.sh --patch` additionally applies the instrumentation
patches under `tools/ref-patches/` (currently: parakeet.cpp tensor-dump
hooks, needed only to regenerate parity dumps). Benchmarks always run stock
pinned references — never benchmark a patched reference.

---

## Results

### M1 Max — full CPU sweep (11 model/format combinations)

`-t 8` both sides, ReleaseFast Fucina vs `llama-bench -ngl 0`, serial runs,
20 prefill lengths + decode per model. Parity band ±3%. Fucina
prefill/decode for the qwen3 models is mean over reps; gemma4/qwen3.5 rows
were best-of only (favorable to Fucina — flagged). Rows sensitive to
measurement conditions — giant-model decode, sweep-tail lengths — use the
stricter prewarmed paired gate (`tools/bench_gate.py`: both engine orders,
per-subprocess cooldowns, `--prewarm-model` page-cache reads); prewarming
is not optional there. A fresh process on a 20 GB model faults the weights
and measures the SSD, not the kernels, and a single cold-cache sample once
read 0.81 on a row that measures 1.06–1.15x prewarmed.

**Headline: of 236 cells — Fucina faster 221, parity 13, llama.cpp faster 2**
(prefill: Fucina 215 / parity 8 / llama 1; decode: 6 / 5 / 1).

Per-model summary (decode is tg64 for dense, tg32 for MoE/Gemma):

| model | prefill geomean (20 lengths) | decode Fucina | decode llama | decode ratio |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B f16 | 1.81x | 83.0 | 80.7 | 1.03 (parity) |
| Qwen3-0.6B Q8_0 | 1.27x | 151.7 | 133.6 | 1.14 |
| Qwen3-0.6B Q6_K | 1.39x | 140.0 | 132.0 | 1.06 |
| Qwen3-0.6B Q5_K_M | 1.65x | 145.5 | 145.8 | 1.00 (parity) |
| Qwen3-0.6B Q5_K_S | 1.47x | 148.4 | 151.7 | 0.98 (parity) |
| Qwen3-0.6B Q4_K_M | 1.45x | 190.9 | 166.3 | 1.15 |
| Qwen3-0.6B Q4_K_S | 1.45x | 198.9 | 173.5 | 1.15 |
| Qwen3-1.7B Q4_K_M | 1.18x | 81.5 | 80.0 | 1.02 (parity) |
| Qwen3.5-0.8B Q8_0 | 1.07x (4 lengths) | 98.6 | 71.9 | 1.37 |
| Qwen3-30B-A3B Q5_K_M (MoE) | 1.77x | 26.3 | 19.4 | 1.36 |
| Qwen3-30B-A3B Q6_K (MoE) | 1.62x | 29.8 | 33.8 | 0.88 (llama; pre-prewarming) |
| Gemma-4-26B-A4B Q6_K (MoE) | 1.47x | 24.1 | 24.7 | 0.98 (parity) |

Protocol notes on the summary values: the Q6_K-0.6B, Qwen3.5, and 30B
Q5_K_M decode cells and the 1.7B/Gemma prefill geomeans use the
cool/prewarmed paired protocol where the plain serial sweep proved
conditions-sensitive; dense decode on M1 is compute- rather than
bandwidth-bound (Q6_K, fewer bytes per weight, read slower hot while Q8_0,
more bytes, read faster — bandwidth would predict the opposite).

The two llama.cpp-win cells, exhaustively: Qwen3.5-0.8B pp32 (0.86) and the
30B Q6_K decode (0.88 — recorded without page-cache prewarming; its Q5_K_M
sibling measures 1.36x under the prewarmed gate). Q6_K-0.6B pp4 and pp7
measure 1.43x and 2.24x under the prewarmed gate (single paired round
each); Qwen3-1.7B pp256 is 1.07x with pp31 confirmed over three prewarmed
rounds (1.06–1.15x).

Representative full tables (all tok/s; ratio = Fucina/llama):

**Qwen3-0.6B Q4_K_S** (a typical dense-format win):

| pp | Fucina | llama | ratio |
|---:|---:|---:|---:|
| 1 | 226.4 | 150.3 | 1.51x |
| 2 | 295.7 | 261.7 | 1.13x |
| 3 | 398.6 | 329.4 | 1.21x |
| 4 | 626.4 | 340.6 | 1.84x |
| 5 | 543.7 | 366.2 | 1.48x |
| 6 | 611.6 | 390.0 | 1.57x |
| 7 | 675.0 | 426.1 | 1.58x |
| 8 | 730.3 | 478.7 | 1.53x |
| 9 | 646.3 | 509.7 | 1.27x |
| 15 | 841.8 | 633.7 | 1.33x |
| 16 | 979.3 | 726.9 | 1.35x |
| 17 | 892.7 | 709.5 | 1.26x |
| 31 | 1025.6 | 856.1 | 1.20x |
| 32 | 1077.3 | 500.4 | 2.15x |
| 33 | 910.7 | 505.7 | 1.80x |
| 64 | 1109.2 | 683.1 | 1.62x |
| 127 | 1129.5 | 839.9 | 1.34x |
| 128 | 1248.8 | 864.4 | 1.44x |
| 129 | 1212.3 | 852.2 | 1.42x |
| 256 | 1219.2 | 936.0 | 1.30x |

decode (tg64): Fucina 198.9 vs llama 173.5 — 1.15x.

Note the llama.cpp discontinuity at pp32 across every dense table: that is
its switch to the Accelerate/AMX BLAS path at batch >= 32, which only
re-amortizes at larger batches.

**Qwen3-30B-A3B MoE Q6_K** (prefill win, decode loss):

| pp | Fucina | llama | ratio |
|---:|---:|---:|---:|
| 1 | 34.3 | 30.2 | 1.14x |
| 2 | 30.5 | 15.6 | 1.96x |
| 3 | 42.4 | 22.1 | 1.92x |
| 4 | 46.8 | 32.0 | 1.46x |
| 5 | 53.2 | 36.1 | 1.47x |
| 6 | 55.2 | 25.9 | 2.13x |
| 7 | 56.1 | 35.8 | 1.57x |
| 8 | 60.4 | 33.9 | 1.78x |
| 9 | 62.3 | 41.2 | 1.51x |
| 15 | 66.5 | 54.4 | 1.22x |
| 16 | 69.4 | 52.2 | 1.33x |
| 17 | 68.9 | 58.8 | 1.17x |
| 31 | 73.3 | 54.4 | 1.35x |
| 32 | 73.2 | 50.5 | 1.45x |
| 33 | 72.1 | 51.0 | 1.41x |
| 64 | 125.7 | 63.8 | 1.97x |
| 127 | 132.8 | 66.1 | 2.01x |
| 128 | 134.8 | 64.6 | 2.09x |
| 129 | 132.4 | 64.1 | 2.07x |
| 256 | 137.1 | 65.7 | 2.09x |

decode (tg32): Fucina 29.8 vs llama 33.8 — 0.88x, **llama.cpp**.

**Gemma-4-26B-A4B MoE Q6_K** — measured with the cool protocol (Fucina:
fresh process per length, best-of, pre-cooldowns; llama.cpp: one combined
warm process per session — fresh per-length llama processes measure far
below llama's real speed on this model and are not used):

| pp | Fucina | llama | ratio |
|---:|---:|---:|---:|
| 4 | 44.6 | 20.8 | 2.15x |
| 6 | 45.1 | 32.6 | 1.38x |
| 7 | 53.4 | 41.2 | 1.30x |
| 8 | 65.5 | 47.9 | 1.37x |
| 9 | 57.2 | 48.8 | 1.17x |
| 15 | 79.3 | 56.3 | 1.41x |
| 16 | 86.2 | 58.4 | 1.48x |
| 17 | 87.0 | 48.8 | 1.78x |
| 31 | 115.2 | 66.5 | 1.73x |
| 32 | 121.6 | 64.1 | 1.90x |
| 33 | 116.0 | 58.8 | 1.97x |
| 64 | 154.1 | 93.0 | 1.66x |
| 128 | 166.4 | 96.9 | 1.72x |
| 256 | 170.2 | 92.5 | 1.84x |

decode (tg32): Fucina 24.1 vs llama 24.7 — 0.98x, parity. pp1–3 are
parity-to-win (1.01–1.19x). The pp64–256 rows come from a separate cool
session in which llama measured at its strongest recorded values.

Disclosure, so the mid-batch rows are not over-claimed: llama's absolute
numbers in the pp4–33 session ran below its strongest recorded values on
this model (e.g. pp16 58.4 vs 82.1 in another session; same binary, model,
and flags — llama's qwen rows reproduced their expected values in the same
session, so binary and machine are fine). The depression has a warming
gradient (earliest lengths hit hardest), consistent with page-cache
pressure penalizing llama's mmap-resident expert access; Fucina is
structurally less exposed because it copies experts into resident packed
buffers at load. Same-session paired rows are the valid comparison and
Fucina wins all of them; conservatively cross-checking Fucina's numbers
against llama's strongest recorded values still gives wins at pp15–33
(86–116 vs 81–97) and leaves pp4–9 dependent on llama numbers that session
could not reproduce. A pristine-machine session (long idle, no prior model
reads) should re-confirm the pp4–9 band before anyone leans on those
specific cells. Ruled out as causes of the mid-batch shape being hard for
both engines: Accelerate/AMX (dequant-sgemm cannot amortize at 1–3 tokens
per expert, and llama.cpp's `mul_mat_id` is int8 there too) and thread
over-subscription (an apparent "4 threads beats 8" was a heat-soak
artifact; cool, 8 > 6 > 4).

### 30B/26B MoE with experts transcoded to Q4_K (a behind-record)

Experts-only Q4_K transcode via the exporter (`zig build export-gguf --
--experts-dtype q4_k`): Qwen3-30B-A3B 25.1→17.6 GB, Gemma-4-26B 23.2→19.2 GB.
Decode follows expert bytes-per-weight (weight-bandwidth-bound): Qwen3-30B
decode Q5_K_M (0.69 B/w) 32.3 tok/s, Q6_K (0.82) 28.5–30.1, Q4_K-experts
(0.56) 34.1 (+20% mean vs same-session Q6_K). Quality: argmax aligned on the
parity prompts, top-5 order approximately preserved.

Same-session llama.cpp comparison (build 30af6e2, `-ngl 0 -t 8 -p 256 -n 32
-r 3`, heat-soaked session — treat as lower bounds both sides):

| model (Q4_K experts) | side | pp256 tok/s | tg32 tok/s |
| --- | --- | ---: | ---: |
| Qwen3-30B-A3B | llama.cpp | 107.44 ± 0.57 | 25.84 ± 0.25 |
| Qwen3-30B-A3B | Fucina | 89.02 ± 2.01 | 26.41 ± 0.84 |
| Gemma-4-26B | llama.cpp | 116.18 ± 3.57 | 19.51 ± 0.34 |
| Gemma-4-26B | Fucina | 98.1 (best-of-3) | 18.7 (best-of-3) |

Net: **Fucina behind llama.cpp on pp256 by ~15–17%** with Q4_K experts
(L2-bound Q4_K prefill), roughly tied on Qwen decode, slightly behind on
Gemma decode in this run.

### Lossless speculative decoding

M1 Max, ReleaseFast, Qwen3-0.6B-Q4_K_S, greedy unless noted. Speedup =
`--spec` vs plain decode tok/s on the same prompt; the committed token
stream is identical by construction (lossless), so this is pure throughput.
Design record and break-even math: `SPECULATIVE.md`.

| Task | Speedup | Notes |
| --- | --- | --- |
| Grounded copy | 1.47x | 70% acceptance, 6.6 tok/step |
| Same prompt, re-encoded post tokenizer fix | 1.04x | 41% acceptance — generation itself diverged |
| Verbatim-repetition microcase | 2.3x | 100% acceptance, ~11 tok/step |
| Code edit | 1.12x | 84.6% acceptance; was 0.79x before the tokenizer parity fix |
| Free-form generation | 0.99x | cost gate auto-off; was 0.83x ungated |
| RAG-injected reference (`--spec-ref`) | 0.98x | 33/68 draft tokens accepted from the reference |
| Grounded copy, sampled t=0.7 | 1.20x | same single-RNG-draw path as greedy |

The cost gate turns speculation off when its rolling speedup estimate drops
below 1.0: worst observed case 0.98–0.99x, vs 14–21% losses ungated.
Tokenizer parity is a precondition for acceptance numbers — the qwen2
pretokenizer is token-ID-exact vs `llama-tokenize` on 14 fixtures.
Context only (not a controlled A/B): llama.cpp's `llama-lookup` reports 93%
acceptance at max-draft 3 on the same grounded text — a shorter-draft
operating point.

### Text-diffusion step time (DiffusionGemma)

M1 Max, 8 threads, ReleaseFast, cool chip, same Q6_K GGUF (22.65 GB) and
threads on both sides. Per denoise step (one 256-token bidirectional canvas
forward + sampler pass): **Fucina ~3.5 s vs `llama-diffusion-cli` (llama.cpp
PR 24423) 4.84 s/step** — ~1.4x. Step counts vary with RNG/template, so
compare seconds/step, not totals. Logit parity sits inside llama.cpp's own
cached-vs-unified numeric spread on this model (the model is numerically
chaotic; small kernel differences amplify).

### x86-64 — Intel i9-13950HX (AVX2 + AVX-VNNI)

First x86 hardware record. No BLAS on either side (verified — see
"Hardware and reference builds"); both engines run their own AVX2/AVX-VNNI
CPU kernels. Fucina numbers are after the packed-VNNI/AVX2 quantized
kernels and the f32-accumulate f16 GEMM landed; "before" is the same tree
without them (the prior kernels fell back to scalar on x86).

Targeted campaign runs (Qwen3-0.6B unless noted; tok/s):

| format | metric | before | after | llama.cpp | verdict |
| --- | --- | ---: | ---: | ---: | --- |
| q8_0 | pp256 | 105 | 761 | 558 | 7.2x self; 1.36x llama |
| q8_0 | pp256, single P-core | 19.1 | 151.9 | — | 8.0x self |
| q4_k_s | pp256 | 164.6 | 801.1 | 631 | 1.27x |
| q4_k_m | pp256 | 174.8 | 713.9 | 607 | 1.18x |
| q4_k_s | decode64 | 54.3 | 92.8 | ~76 | win (was 0.72x) |
| q4_k_m | decode64 | 55.7 | 87.1 | ~76 | win |
| q6_k | pp256 | 120.7 | 531.0 | 385 | 1.38x (was 0.31x) |
| q6_k | decode | 63.2 | 67.4 | 68.4 | 0.986x — parity |
| q5_k | decode | — | ~64 | 79.8 | **0.80–0.90x — llama.cpp wins** |
| Qwen3-30B-A3B Q6_K (MoE) | pp256 | 18.07 | 24.06 | 53 | still behind pre-rerun; see matrix below |
| f16 model | pp1024 | 17.9 | 354 | — | ~20x self |
| f16 model | tg | 9.3 | 28.5 | — | 3.1x self |

The full paired-gate matrix run the same day gives the durable medians
(paired orders, median-of-samples — generally lower than the targeted rows
above on both sides; e.g. q8_0 pp256 681.7 vs 468.9, ratio 1.45; Q5_K_M
pp256 677.7 vs 298.2, ratio 2.27; Q5_K_S pp256 729.6 vs 294.1, ratio 2.48).
Matrix verdicts per model (PASS = ratio >= 1.0 with CV <= 8% both sides;
NOISY = excessive variance, not a result; FAIL = Fucina behind):

| model | pass/noisy/fail | ratio min/med/max | residual FAILs |
| --- | --- | --- | --- |
| qwen3-0.6b q8_0 | 8/13/0 | 1.04/1.32/1.95 | — |
| qwen3-0.6b q4_k_m / q4_k_s | 13/8/0 · 8/13/0 | 1.21/1.46/1.93 · 1.04/1.40/1.82 | — |
| qwen3-0.6b q6_k | 14/6/1 | 0.99/1.41/1.66 | decode 0.987 (coin-flip parity) |
| qwen3-0.6b q5_k_m / q5_k_s | 10/10/1 · 14/6/1 | 0.85/1.70/2.29 · 0.90/1.95/2.56 | decode 0.90–0.91 (m=1 GEMV path, open) |
| qwen3moe-30b q5_k_m | 5/0/16 | 0.46/0.80/1.30 | mid-batch pp15–33 (0.46–0.52) + decode 0.90 |
| qwen3moe-30b q6_k | 2/0/19 | 0.36/0.76/1.11 | mid-batch pp15–33 (0.36–0.43) + decode 0.95 |
| gemma4-26b q6_k | 12/1/8 | 0.85/1.04/1.31 | small-batch pp2–9 (0.85–0.98) |

Reading: the dense 0.6B is swept (large-batch ratios 1.3–2.6x). The 30B MoE
wins or holds parity at pp64+ on Q5_K_M (1.24–1.30x) but **loses mid-batch
prefill decisively** — at 128 experts / top-8, pp8–33 gives each selected
expert only ~1–3 rows, so each expert's full weight matrix streams from
memory for almost no work (a weight-bandwidth shape llama.cpp's
`mul_mat_id` handles better) — and m=1 decode stays bandwidth-bound at
0.90–0.95x. Gemma on x86: llama.cpp wins pp1–9 (0.85–0.99x), Fucina wins
pp15+ (up to 1.31x), decode parity.

Numerics: all quantized x86 kernels are i32-bit-exact vs their scalar
reference arms; the f16 change (f32 accumulation) is strictly more accurate
than the old per-op rounding, proven bit-identical on a Q8_0 model forward.

**Update 2026-07-10 — the MoE mid-batch band is resolved.** Two coupled
scheduler changes (same llama build 30af6e2 reference, same flags, `-t 8`,
prewarmed, paired orders, reps 3): the batched-MoE phased chain now engages
at 64 routed pairs instead of 512 (at top-8 the old gate excluded everything
below seq 64, so the pp15–33 band ran one monolithic task per expert), and
small-m (m < 16) projection phases column-chunk at 256 when the layer has
fewer than workers x 16 active experts. Decomposition on the same day, same
box (tok/s Fucina vs llama, ratio):

| variant | Q5_K_M pp15 / pp17 / pp32 | Q6_K pp15 / pp17 / pp32 |
| --- | --- | --- |
| old gate (512 pairs) | 0.509 / 0.500 / 0.461 | 0.465 / 0.441 / 0.375 |
| chain at 64 pairs only | 1.131 / 1.162 / 1.207 | 0.940 / 0.958 / 0.963 |
| + small-m column chunking | 1.152 / 1.175 / 1.218 | 1.050 / 1.021 / 0.987 |

The chain gate does the heavy lifting; chunking adds ~2% on Q5_K_M and
+2.5–11% on Q6_K — Q6_K needs both to cross parity at pp15–17. Full-band
run (pp15,16,17,31,32,33): Q5_K_M 6/6 PASS at 1.15–1.22x; Q6_K passes
pp15–17 (1.02–1.05x) and sits at 0.965–0.987x at pp31–33 (residual gap,
recorded under Open gaps). On the M1 the same change is +27–37% self-A/B
across the band (68.3→86.5 tok/s at pp15, 75.7→103.5 at pp32; interleaved
OLD/NEW, 2 passes x 3 reps, sigma under 3.3). Routing skew observation from
the profile counters: at pp256 only ~52 of 128 experts are active per batch
(uniform routing would predict ~124), so the small-m heuristic also engages
beyond the target band.

**Update 2026-07-10 — Q5_K dense decode resolved (measured; landing queued
behind in-flight tree work).** The 0.90x loss was layout bytes, not
kernels: every dense Q5_K weight is resident twice (`WeightQ5_K
{value, packed_rhs}`), and decode streamed the prefill-favorable
byte-expanded `Q5_Kx8` pack at 8.625 bpw where the GGUF-native blocks in
`value` are 5.5 bpw — 1.57x the necessary bytes at a DRAM-bound m=1 GEMV
(llama.cpp's x86 arm does not even repack Q5_K; it wins on density alone).
Routing m &lt; 4 to the already-resident compact blocks through the existing
tensor-RHS kernels — a ~30-line dispatch gate, zero added memory, proven
bitwise-identical cross-layout at m=1..3 by a new kernel test — measured on
the i9: decode Q5_K_M 0.90 → **1.016 PASS**, Q5_K_S 0.91 → **1.055 PASS**
(self-A/B 66.1 → 79.0 tok/s, +19.5%), prefill 7/7 PASS including the
compact-routed pp1–3 (1.10–1.32), 30B MoE decode ride-along 0.90 → 0.972
(dense tensors route compact; the expert residual remains). The Q6_K twin
gate (1.30x expansion) measured the proportional result: dense decode
0.987 → **1.057 PASS** (self-A/B 66.7 → 74.0 tok/s, +10.9%), 30B MoE Q6_K
decode 0.93 → 0.971. Both gates are bitwise-proven (kernel cross-layout
tests are on the tree) and are landed.

Knob sweep and regression legs (same day, same protocol): raising the
task-budget multiplier to 32 does NOT move the Q6_K pp31–33 residual
(0.952–0.979 vs 0.965–0.987 at 16 — noise-indistinguishable), so the
shipped constants stay gate=64 / chunk=256 / budget=workers x 16 and the
residual is attributed to the Q6_K packed layout's 1.30x byte expansion at
small m, not scheduling. Broad sweep pp1–256 + decode on both 30B quants:
no regressions (Q5_K_M pp64–256 wins hold at 1.24–1.28x; Q6_K pp64–256 now
0.93–0.97x, up from a 0.76 median / 0.45x targeted pp256 record; pp1–7
stays the known monolithic-path band; decode unchanged as expected — the
seq==1 path is untouched). Gemma-4-26B full leg: non-regressed (pp2–9
stays its pre-existing 0.91–0.97 band, pp15–64 wins hold, decode parity
0.996).

### Batch-N multi-stream decode (M1 Max)

`--streams N` on the qwen3 runner: N lockstep decode streams (per-stream KV
cache and sampler, one m=N weight pass per step) vs the same binary running
the N generations back to back.

```sh
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  <8-token-ids> --gen 65 --bench 3 --streams 4
```

M1 Max, ReleaseFast, 8 threads, cool machine, Qwen3-0.6B, 8-token prompt,
64 decode steps per stream:

| weights | N | batch tok/s | sequential tok/s | speedup | outputs |
| --- | --- | ---: | ---: | ---: | --- |
| Q4_K_S | 2 | 152.6 | 136.2 | 1.12x | identical |
| Q4_K_S | 4 | 313.5 | 121.6 | 2.58x | ~1e-6 logit drift (packed 4-row kernels) |
| Q4_K_S | 8 | 369.7 | 116.0 | 3.19x | ~1e-6 drift |
| Q6_K | 2 | 139.8 | 101.2 | 1.38x | identical |
| Q6_K | 4 | 254.1 | 92.8 | 2.74x | identical |
| Q6_K | 8 | 358.3 | 110.2 | 3.25x | identical |
| f16 | 4 | 205.5 | 66.4 | 3.09x | identical |

Decode is weight-bandwidth-bound, so batching N streams into one m=N pass
reads the weights once instead of N times. The modest N=2 gain is the
per-row (m<4) quantized kernels re-streaming the packed weights; the
weights-read-once kernels engage at m>=4 — which is also the documented
~1e-6 numerics boundary for quantized weights (f32/f16 stay bitwise,
verified to m=12).
Single-stream non-regression was verified with interleaved cool A/B pairs
(prefill neutral; decode differences within thermal-order noise, both signs
observed).

### Parakeet ASR vs parakeet.cpp (M1 Max)

(Parity-dump regeneration for this family needs the instrumentation patch:
`tools/fetch_refs.sh --patch parakeet.cpp`. The benchmark rows below were
measured against the stock pinned binary.)

A different reference engine (ggml-based, Accelerate), same discipline. NeMo
Parakeet FastConformer models; transcripts are **byte-identical** to
parakeet.cpp in every mode below, so these are pure-speed comparisons.
Fixture: speech.wav, 7.435 s. The port started ~7–8x slower than
parakeet.cpp and closed the gap in stages; final states:

Cold one-shot (full process incl. load, best/median of 7, tdt_ctc-110m,
measured cool):

| format | Fucina best/median | parakeet.cpp best/median | result |
| --- | ---: | ---: | ---: |
| f16 | 142.6 / 143.9 ms | 156.2 / 159.4 ms | 1.09x faster |
| q8_0 | 136.9 / 137.6 ms | 143.9 / 148.7 ms | 1.05x faster |
| q6_k | 173.1 / 177.2 ms | 188.3 / 190.5 ms | 1.09x faster |
| q5_k | 167.1 / 169.1 ms | 182.1 / 185.2 ms | 1.09x faster |
| q4_k | 148.0 / 150.8 ms | 162.0 / 164.5 ms | 1.09x faster |

Warmed steady-state session (`--f32-cache --fast-mel`, 20 reps, best; vs
`parakeet-cli bench`): q8_0 97.5 vs 102.230 ms (1.05x), q6_k
99.5 vs 144.659 (1.45x), q5_k 103.3 vs 144.615 (1.40x), q4_k 102.4 vs
123.898 (1.21x). **f16 is not a win in this mode** (Fucina 101.6 vs 93.404
ms) — the big dense f16 GEMM is where ggml's Accelerate/AMX path is
strongest.

Streaming head-to-head (per-invocation wall incl. load, warm page cache,
best-of-6, 8 threads): realtime_eou-120m 667 vs 846 ms (Fucina
1.27x faster); nemotron-0.6b multilingual 1309 vs 1675 ms (1.28x faster);
offline tdt-0.6b-v3 567 vs 532 ms (0.94x — parakeet.cpp edges it, same
AMX-favored dense-GEMM story).

### OmniVoice TTS vs omnivoice.cpp (M1 Max)

The OmniVoice port is validated and benchmarked against omnivoice.cpp, the
C++ reference it ports: MaskGIT token streams and RVQ codes are
**byte-exact** at F32 with a fixed seed, decoded audio cosine >= 0.99999,
and synthesis runs **2.3–4.6x faster** than omnivoice.cpp's CPU backend at
equal dtypes (32x on BF16, where ggml's CPU BF16 matmul path is
pathological — use Fucina's BF16, not the reference's, if you want that
dtype). Setup, per-dtype guidance, and the determinism contract:
`examples/omnivoice/README.md`.

### LocateAnything-3B detection vs locate-anything.cpp (M1 Max, 2026-07-07)

Open-vocabulary detection VLM (MoonViT + Qwen2.5-3B + MTP parallel box
decoding) vs the ggml-based reference it ports (mudler/locate-anything.cpp @
92c1682, CPU backend). Detections are **byte-identical** in every cell below
(the JSON outputs `cmp` equal), and the underlying token streams are id-exact
(the `compare` gate battery), so these are pure-speed comparisons.

Protocol mirrors the reference's own benchmarks/BENCHMARK.md: the 448x448
parity fixture, prompt "cat</c>remote", greedy, full streams (early-stop off
on both sides), 8 threads, warm, wall of `detect` with load subtracted via
`info` (ref ~3.9 s f32 / ~1.0 s q8; ours ~2.7 / ~1.2 s). Two interleaved
passes each; both shown:

| mode | dtype | locate-anything.cpp | Fucina | inference speedup |
| --- | --- | ---: | ---: | ---: |
| slow (pure AR) | f32 | 22.9 / 24.7 s | 10.6 / 11.8 s | **~2.3x** |
| hybrid (PBD, default) | f32 | 44.3 / 45.4 s | 29.4 / 35.5 s | **~1.4x** |
| fast (MTP-only) | f32 | 41.5 / 40.6 s | 26.3 / 26.7 s | **~1.6x** |
| slow | q8_0 | 10.9 / 11.2 s | 5.5 / 5.4 s | **~2.4x** |
| hybrid | q8_0 | 18.4 / 18.6 s | 9.3 / 8.9 s | **~2.2x** |
| fast | q8_0 | 18.2 / 17.9 s | 9.5 / 8.6 s | **~2.2x** |

q8_0 boxes are additionally byte-identical to each engine's own f32 output —
stronger than the cross-implementation guarantee the tolerance policy
promises for quantized paths, recorded as a fixture-level observation, not a
contract.

The hybrid/fast gap was closed by the `gemmNTCols` loop interchange (see
*Recorded negatives* below and the commit): the MTP block rounds are
m = n_recompute + 6 ≈ 8-row NT GEMMs, squarely in the f32 column-kernel
window whose row-outer loop re-streamed the whole RHS per row. Routing those
shapes to Accelerate instead was measured slower (37.8 s vs 32.5 s wall,
fast mode) — the fixed vector kernel wins tall-skinny small-m on this
machine.

**x86-64 leg (i9-13950HX, 8P+16E, no BLAS either side, 2026-07-07).** Same
fixture/protocol, both engines compiled on the box (gcc-12 / zig native).
The full parity battery re-ran green on x86 (39 gates, ReleaseFast — the
AVX2/AVX-VNNI kernel arms included); the remote `zig build test` suite
passed natively first. At 8 threads pinned to the P-cores (`taskset -c
0-15`, median of 3, wall; f32 boxes byte-identical per mode):

| mode | dtype | locate-anything.cpp | Fucina | inference speedup |
| --- | --- | ---: | ---: | ---: |
| slow | f32 | 26.1 s | 19.1 s | **~1.4x** |
| hybrid | f32 | 40.3 s | 30.0 s | **~1.4x** |
| fast | f32 | 37.5 s | 28.4 s | **~1.3x** |
| slow | q8_0 | 13.7 s | 11.2 s | **~1.3x** |
| hybrid | q8_0 | 22.3 s | 19.1 s | **~1.2x** |
| fast | q8_0 | 21.8 s | 18.5 s | **~1.2x** |

Honest reads on the x86 leg:

- **16 threads is parity-to-behind, and HT-pinned is a Fucina pathology.**
  Unpinned t16 (E-cores in play) is noisy parity on f32 and ~0.9x on q8
  hybrid/fast (ggml slightly ahead). Pinning 16 Fucina workers onto the 8
  P-cores' 16 hyperthreads collapses our side (slow 19 -> 43 s, hybrid
  30 -> 72 s) while ggml holds its t8 level — the spin-then-park worker
  team degrades under HT oversubscription where ggml's threading does not.
  The 8-thread numbers above are the intended operating point.
  **Resolved 2026-07-10:** `cpuThreadCount` now min()s in the physical-core
  count (Linux: `thread_siblings_list` dedup intersected with the affinity
  mask; macOS: `hw.physicalcpu`, which equals the logical count on all
  Apple Silicon so no mac config changes). Verified on this box: unpinned
  sizing 32→24, `taskset -c 0-15` (this collapse scenario) → 8, a mixed
  2P+4E mask → 6; `FUCINA_MAX_THREADS`/`setMaxThreads` semantics preserved
  (`setMaxThreads` still deliberately oversubscribes past physical).
- **q8_0 boxes on x86 are not cross-engine identical** (they are on the M1):
  activation-quantization rounding differs per kernel arm. Real detections
  match in every mode; slow mode drifts two coordinates by one coordinate
  token (~0.45 px), hybrid only inside the degenerate repeated-box tail;
  fast is byte-identical. Reference-q8-vs-its-own-f32 is 0 fields on this
  fixture, ours-q8-vs-our-f32 is 2 (slow) / 25-in-tail (hybrid) / 0 (fast)
  — within the tolerance policy's expectations for quantized paths, and the
  f32 streams stay byte-identical cross-engine on both ISAs.

### Fucina-only kernel context (no llama.cpp pairing)

Two internal records that frame the numbers above:

- **Query-tiled online-softmax attention forward**: M1, Qwen3-0.6B —
  attention phase ~2.1x; end-to-end prefill pp1024 +24.5%, pp2048 +56%,
  pp4096 2.19x; no regressions across the routine matrix.
- **Blocked packed f32 GEMM for the no-BLAS build** (`-Dblas=none`, the
  default off macOS): 2048^3 109 → 608 GFLOP/s (5.6x), reaching ~26–35% of
  Accelerate/AMX on the same machine — what makes the no-BLAS builds
  credible at training shapes.

---

## Recorded negatives

Kept on record so they are not re-tried or over-claimed:

- **BLAS at small-m tall-skinny f32 NT shapes loses to the fixed vector
  column kernel (M1 Max).** Lowering `shouldUseBlas` to m >= 2 routed the
  LocateAnything MTP-round GEMMs (m ≈ 8, n up to 11008/152681, k = 2048) to
  Accelerate and measured 37.8 s fast-mode wall vs 32.5 s for the
  interchange-fixed `gemmNTCols` (2026-07-07). The m >= 16 BLAS gate stays;
  do not re-lower it without new evidence on these shapes.

- **Dispatcher-QoS pin did not fix the qwen3.5 pp32 variance.**
  The barrier workers already elevate to `QOS_CLASS_USER_INTERACTIVE`; the
  dispatcher thread (which computes chunk 0 of every parallel op) and the
  dot-backward worker did not. Pinning them too is kept — it restores
  QoS parity across all compute threads, matches llama.cpp's practice, and
  measured no regression — but it did NOT collapse the qwen3.5 pp32
  process-to-process bimodality (still 446–667 tok/s best-of-3 across six
  fresh processes afterwards), so the root cause of that spread remains
  open.
- **Q5_K decode repack: tried and reverted.** A repack that helped decode
  regressed prefill; the tree keeps the prefill-favorable layout.
- **q8_0 KV cache is a capacity option, not a speed option**:
  1.88x context in the same cache budget, but decode is *slower* with it on
  M1 — decode attention there is compute-bound, and the dequant adds ~2.3x
  to the attention phase at 2048 ctx.
- **bf16-resident weights buy memory, not speed, on M1**:
  decode 54.7 tok/s vs f16's 65.5; the M1 has no bf16 FMA, so the kernel
  widens in-register.
- **Residual-add epilogues: measured and declined (2026-07-10).** The
  profile counters at the qwen3 residual sites put the entire opportunity
  at 0.7% (decode) / 1.2% (pp256) of forward time on the 0.6B and 0.1% /
  0.34% on the 30B MoE — the capturable slice of an in-place/consuming-add
  rewrite (roughly half) sits inside paired-gate noise, and a true beta=1
  GEMM epilogue cannot be proven bit-exact for BLAS (the CBLAS contract
  fixes the value, not the summation order). Do not re-try without a
  workload where the residual share is measured above ~2%.
- **trans_ab contraction kind: measured and declined (2026-07-10).** The
  role-pinned double-transposed GEMM — the only case where the einsum engine
  materializes an operand — has zero reachable sites (every production
  dot/einsum call and every training-backward einsum verified to resolve to
  plain/trans_a/trans_b), and the measured recovery ceiling is 1.13-2.8x
  (M1, Accelerate) in a case nothing hits. The equation-level role-swap
  workaround is faster than the trans_ab ceiling itself (Accelerate's NN
  beats its TT at every measured shape), and on no-BLAS builds at most
  1.06-1.32x is recoverable — likely unattainable with a TT access pattern.
  Re-open trigger: a port whose profile shows the tagged.zig materialize arm
  hot with genuinely pinned output; the recorded BLAS-arm-only design
  (~250 LOC) then applies, and the cheaper first lever is a blocked
  transpose in the materialize pass (currently ~3-8 GB/s).
- **Open llama.cpp gaps** as of the latest records: Gemma-4-26B mid-batch
  prefill on M1 (0.72–0.91x cool, cause unresolved); Qwen3-30B MoE Q6_K
  pp31–33 on x86 (0.965–0.987x residual of the resolved mid-batch band —
  byte-expansion-attributed, see the two 2026-07-10 updates) and MoE decode
  (x86 0.971–0.972 after the compact ride-alongs; M1 0.87–0.95); pp256 with
  Q4_K experts on M1 (~0.85x). x86 dense decode on Q5_K and Q6_K is
  resolved (1.016/1.055 and 1.057; gates landed).

## Limitations

These results are **shape- and machine-specific**: two CPUs (Apple M1 Max,
Intel i9-13950HX), one thread count each, specific GGUF files, specific
prompt lengths. On Apple Silicon, thermal state changes outcomes by tens of
percent — several conclusions in this file flipped between hot and cool
measurements, and only cool, isolated, best-of A/B pairs proved
trustworthy. Fucina's benchmark loops still include final logits/sampler
work that llama-bench skips, which biases every ratio here *against*
Fucina by a small amount. And the comparison target moves: llama.cpp
advances continuously, so a dated ratio is a statement about two specific
builds on one day, not a standing property. Re-run the paired gate on your
own hardware before relying on any row.
