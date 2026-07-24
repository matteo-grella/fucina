# PTQTP — Post-Training Quantization to Trit-Planes

Design record. Data-free post-training quantization of weight matrices into
K ∈ {1,2,3} ternary planes over the TQ2_0 machinery (docs/TERNARY.md).
Method: arXiv:2509.16989 (Xiao et al.), implemented from the paper's
formulas; K = 3 is a Fucina capacity extension beyond the paper's dual
decomposition. No calibration data, no retraining, no gradients: weights
in, packed trit-planes out.

## The method

Each weight matrix decomposes as `W ≈ Σₖ diag(αₖ)Tₖ` with trit planes
`Tₖ ∈ {-1,0,+1}` (topology) and one scale per plane per length-256 column
group (magnitude). Each group is an independent problem, solved by
alternating:

1. **scales** — closed-form K×K ridge regression `α = (SᵀS + λI)⁻¹Sᵀw`
   (Gram entries are integer trit counts). λ escalates ×10 from `lambda0`
   (1e-8) while the Frobenius condition estimate exceeds `kappa_max` (1e6),
   clamped at `lambda_max` (1.0) — required at init, where all planes start
   at sign(w) and the unregularized system is singular.
2. **topology** — per-element exhaustive 3ᴷ-way argmin of
   `(w − Σₖ αₖcₖ)²` over trit tuples.

A group converges when its scale vector moves less than `epsilon` (1e-4),
capped at `max_iterations` (50). The candidate order is pinned — zero tuple
first, sparser before denser, finer planes first, strictly-less
keeps-first — so exact ties prefer sparser trits, the symmetric init breaks
deterministically, and the whole solve is bitwise reproducible for any
thread count (rows fan out with disjoint outputs; per-row stats reduce in
row order).

Deliberate deltas from the paper:

- **G = 256, packed.** The paper uses G = 128 unpacked. Fucina's group size
  is the TQ2_0 block width, so each plane is a byte-valid standalone TQ2_0
  tensor whose per-block fp16 `d` IS the group scale: inference is K stock
  ternary matmuls plus adds, no new kernels, and every plane is
  individually llama.cpp-dequantizable. Measured cost of the coarser
  groups: ~1.3% relative error (`reconstructReference` runs any G for such
  studies).
- **fp16 scale rounding at pack time.** |α| is rounded to fp16 first, then
  one final topology pass runs against the rounded scales — stored trits
  are elementwise-optimal for exactly the scales inference multiplies by
  (measured: packed error equals the f32-scale reference to 4 decimals).
  Packing |α| loses nothing: the candidate set is sign-symmetric.
- **Non-finite weights** are excluded from the scale regression and forced
  to trit 0 in every plane — one NaN degrades only itself.
- **K = 1** is a least-squares upgrade over the blind absmean b1.58
  encoder; **K = 3** adds a residual plane (27 levels per group, error
  bound ~1/3 of dual) at +2.06 bpw where applied. Planes are separable by
  construction: serving fewer planes than were solved is valid.

## Surfaces

- `src/ptqtp.zig` (`fucina.ptqtp`): `solveGroup` (pure, allocation-free),
  `quantizeMatrix` → `PlanePair` (owns up to three `[]BlockTQ2_0` planes,
  borrowed `rhs(plane)` matmul views, `reconstructInto`, `MatrixStats` —
  rel Frobenius error of the served reconstruction, per-plane zero
  fractions, iteration/convergence counts), `reconstructReference`
  (arbitrary G, f32 scales, unpacked; the fidelity-study path). Sibling
  tests pin: exact ternary recovery, byte parity with
  `quantizeRowTQ2_0ScaledInto` (the layout contract), error ordering
  K3 < K2 < K1 < absmean, RHS-view matmul equivalence, NaN benignity,
  determinism, all-zero packing, option/shape validation.
- `src/llm/weights.zig`: the `LinearWeight` union has a
  `ptqtp: WeightPtqtp` arm (up to three plane tensors).
  `LinearWeight.toPtqtp` dequantizes rows in chunks through `getRowsAs` —
  **any loadable source dtype quantizes through one code path** (f32, f16,
  bf16, K-quants, legacy, cold formats) — packs the planes, and drops the
  original storage. `ptqtpEligible` gates on the 256-block contract
  (in-dim % 256 == 0). `getRowsAs` on the arm returns the dequantized
  plane sum, so `toResidentF16` doubles as un-decorate. Both LLM trainers'
  frozen-dot dispatch handles the arm (per-plane frozen dots + adds).
- **Fused inference entry** (`linearSeqPtqtpFused`, the `linearSeq` fast
  path for the arm): one Q8_K activation quantization and ONE worker-team
  dispatch per decorated linear; column-partitioned tasks compute every
  plane and sum in the fixed plane order. Bitwise identical to a per-plane
  facade dot chain (pinned by test). Falls back to facade dots for
  gradient-tracking or non-contiguous inputs. Dispatch granularity is the
  decode bottleneck this design addresses: per-plane fork-joins cost more
  than the ternary kernel itself at 1.7B decode shapes.
- `src/llm/qwen3/model.zig`: `Model.decoratePtqtp(ctx, options)` walks
  attention q/k/v (split or fused), o_proj, and dense FFN projections in
  place. `DecoratePtqtpOptions`: `solver` (plane count etc.),
  `skip_first_layers`/`skip_last_layers` (edge layers stay in source
  precision — pure configuration, still data-free), `down_planes`/
  `o_planes` (per-projection plane-count overrides for the sensitive
  residual-writing projections). Embeddings, lm_head, and norms are not
  walked; MoE FFNs are counted skipped.
- **GGUF persistence** (`src/llm/ptqtp_gguf.zig`): a decorated model saves
  as one byte-valid standalone TQ2_0 tensor per plane — `<name>.ptqtp0/1/2`
  replaces `<name>`, each plane individually llama.cpp-dequantizable — plus
  a `fucina.ptqtp.version` metadata key; every other tensor and metadata
  entry passes through byte-verbatim. Loading pair-detects per tensor
  (metadata gate first, so undecorated files pay one map lookup; skip-layer
  tensors inside a decorated file fall through to their base) and rebuilds
  the `.ptqtp` arm bitwise. Fused in-memory weights persist under their
  SOURCE tensor names — the solver's per-group independence makes the
  fused rows' planes byte-identical to solo decoration, so they row-slice
  out losslessly — and re-fuse at load through `fuseLinear`'s ptqtp arm.
  `Model.savePtqtpGguf` walks the projections `decoratePtqtp` covers plus
  the output head; save→load→save is byte-stable. The format invariants —
  plane replacement, fused row-slicing vs solo decoration, 3-part
  re-fusion, resave stability, save/load validation errors — are pinned
  by sibling tests (`ptqtp_gguf_tests.zig`). Decoration thus runs once
  (`--save`): the saved file serves through the ordinary qwen3 runners —
  chat CLI, speculation, batch — with no re-decoration. Pair-detection is
  wired in the qwen3 loaders only; other families do not read decorated
  files yet.
- **Scale-tied fit** (`Options.tie_scales`, `--tie-scales`): locks the plane
  scales to the exact ratio 3 (`[3s, s]` at K=2, `[9s, 3s, s]` at K=3),
  making the K planes one uniform symmetric 3^K-level quantizer — the
  precondition for folding all K planes into a single dot pass
  (`c = 3t₁+t₂`, exact algebra). Measured on Qwen3-0.6B (512-token
  teacher-forced NLL vs the f16 baseline ppl 71.98): K=2 free ppl 210.5 vs
  **tied 203.9**; K=3 free ppl 80.7 (35.7 s fit, 10,497 unconverged groups)
  vs **tied 75.9** (7.0 s, 0 unconverged) — the tied fit reconstructs
  slightly worse (uniform levels, fewer degrees of freedom: rel err .1887
  vs .1784 at K=2) yet measures no downstream quality loss, fits 1.5-5x
  faster, and always converges. Single-eval-text caveat; a second
  model/eval confirmation is the gate for making it the default. The
  per-plane f16 scales round independently, so folded execution must
  derive the coarse scales from the finest in f32 (exact), not re-read
  the rounded pair.
- **Metal prefill offload** (`-Dgpu=metal`): `WeightPtqtp.init` also copies
  each plane into GPU-resident bytes, and prefill-sized fused linears
  (m ≥ 32, work ≥ `FUCINA_GPU_MIN_WORK_DENSE_TQ2`, default 2^25) dispatch
  each plane through the ternary dequant-in-kernel `mul_mm`
  (`fucina_mul_mm_tq2_0_f32`), summing the K plane outputs on the CPU. Not
  bitwise vs the CPU chain — the same accepted numerics stance as the
  Q4_K/Q6_K/Q8_0 dense offload; provider parity tests pin the kernel.
  Measured (M1 Max, Qwen3-0.6B, pp1001, same binary, `FUCINA_GPU=0` as the
  CPU arm): prefill 2830 → 1291 ms at K=2 with an f16 head (**2.2×**), and
  2841 → 956 ms fully ternary with `--head-planes 2` (**2.97×**, ~1047
  prefill tok/s). Decode never dispatches (m ≥ 32 gate). Known follow-up:
  the K plane dispatches sync per linear for the CPU sum — the shared-input
  batch entry can fold them into one command.
- **Runtime speed path**: `WeightPtqtp.init` packs every plane into the x4
  column-interleaved form at construction (`BlockTQ2_0x4`, docs/TERNARY.md
  — same bytes rearranged, zero per-block reduces), and the fused linear
  runs all K planes in ONE worker-team dispatch on the x4 kernels, the
  accumulating twin folding each extra plane straight into the output with
  no scratch pass. Bitwise equal to the per-plane facade dot chain (pinned
  in `weights_tests.zig`). Measured on M1 Max (Qwen3-0.6B, paired
  same-window decode runs): **+14% at K=2 and +30% at K=3** over the
  pre-pack fused path. Odd `n` or unreadable plane storage falls back to
  the row kernels — identical bits, just slower.
- **MoE expert stacks** (`MoeRhs.ptqtp`, `src/exec/moe.zig`): expert
  stacks quantized at K=2/3 run through the fused MoE ops — the tile dot
  runs the ternary kernel once per plane and SUMS per element in fixed
  plane order before the gated nonlinearity, so a PTQTP expert equals the
  dense fused linear bitwise on the same weights (decode GEMV and batched
  prefill both; pinned against host-side per-plane K=1 sums in
  `expert_store_tests.zig`). Persistence reuses the dense convention —
  `<name>.ptqtpK` siblings with the base stack's 3D shape, plane-major on
  disk — and the streamed tier gathers one expert's K plane row-blocks
  into a contiguous cache-slab section (`ProjSpec.plane_count`/
  `plane_offsets`), so K-plane expert models stream out-of-core with no
  new kernels. Loaders: `ptqtp_gguf.maybeLoadMoeRhs` /
  `maybeStreamedMoeProjSpec`, wired into the qwen3 MoE load paths.
- Examples: `zig build ptqtp-spirals` (self-verifying acceptance demo:
  float-trains an MLP, decorates post-training, PASSes only if dual planes
  hold accuracy on the deployed int8 path) and `zig build ptqtp-qwen3`
  (decorate a GGUF in place: `--planes 1|2|3`, `--down-planes/--o-planes N`,
  `--skip-first/--skip-last N`, `--head-planes N` to decorate the lm_head,
  `--nll FILE` teacher-forced perplexity before/after, `--save FILE` to
  persist the decorated model, greedy completion + decode timing).

## Shard-streaming quantizer (`zig build export-gguf -- --ptqtp`)

> The copy-paste walkthrough with measured end-to-end numbers (quantize →
> run resident → run streamed) is `PTQTP-RECIPE.md`; this section is the
> tool reference.

`ptqtp-qwen3 --save` decorates a loaded model — it needs the model in RAM
and knows only the qwen3 family walk. The export tool's `--ptqtp` mode is
the scale path: it quantizes GGUF→GGUF **one source tensor at a time**, so
a hundreds-of-GB model (DeepSeek/GLM class) quantizes on a 64 GB machine.

```sh
zig build export-gguf -Doptimize=ReleaseFast -- \
  --from-gguf big-model-BF16.gguf --out big-model-ptqtp3.gguf --ptqtp=3
```

| Flag | Meaning |
|---|---|
| `--ptqtp[=K]` | enable the mode; `K` = plane count 1–3 (default 2) |
| `--ptqtp-planes K` | plane count as a separate knob (implies `--ptqtp`) |
| `--ptqtp-include SUB[,SUB]` | quantize only tensors whose name contains a substring (replaces the default embeddings/head-stay name policy); repeatable |
| `--ptqtp-exclude SUB[,SUB]` | never quantize matching tensors; always subtracts; repeatable |
| `--dry-run` | print the per-tensor plan (name, shape, source dtype → target, bytes before/after) and exit without writing |

Mechanics and policy:

- **Streaming both ways.** The source is mmap-loaded (split GGUFs load via
  `loadMmapAuto`; the single-file output drops the `split.*` markers), and
  the output uses the writer's streaming path (`gguf.Writer.declareTensor`
  + `beginStream`): header first, then per tensor decode → solve → write →
  release. The tool never holds more than one tensor's f32 buffer plus its
  packed planes (reported as `peak tensor working set`; a 0.6B run peaks at
  ~14 MiB heap). Source pages get `MADV.DONTNEED` after each tensor —
  released immediately on Linux; Darwin ignores the hint for file-backed
  maps and evicts only under pressure, so macOS peak RSS includes clean
  evictable mmap pages (annotated in the summary). MoE expert stacks are
  the one deliberate exception to one-tensor residency: their K plane
  stacks accumulate in RAM for the stack's duration (see below).
- **Same on-disk format as `ptqtp_gguf.zig`**: eligible matrices are
  replaced by `<name>.ptqtp0..K-1` byte-valid TQ2_0 plane tensors plus the
  `fucina.ptqtp.version` stamp, so outputs load through the existing
  pair-detection with zero loader changes (wired in the qwen3 loaders
  today — see the persistence bullet above; other families read the format
  once their loaders adopt the same seam).
- **Eligibility**: 2D matrix or 3D `*_exps` expert stack, name ends
  `.weight`, no `norm`, contract dim divisible by 256, source dtype
  decodable (f32/f16/bf16/legacy/K-quants; quantized sources dequantize
  first — the from-quantized path degrades gracefully, see above). Default
  name policy keeps embeddings (`token_embd`) and `output.weight` in
  source precision; `--ptqtp-include` replaces it (e.g. to decorate the
  head).
- **MoE expert stacks (3D) quantize per expert slice**
  (`ptqtp_gguf.quantizeMoeStack`): each expert's `[out x in]` matrix of
  the expert-major `[in, out, n_expert]` stack runs through
  `ptqtp.quantizeMatrix` independently — group independence makes every
  expert row-block byte-identical to decorating that expert alone — and
  the K plane tensors keep the base 3D shape, plane-major (the exact MoE
  convention the qwen3 loaders pair-detect, resident and streamed).
  Memory: the K accumulating plane stacks stay resident for the whole
  stack plus one expert's f32 slice (~550 MiB per plane for a
  4096 x 2048 x 256-expert stack, ~1.7 GiB at K=3); both the dry-run plan
  and the run summary report the figure, and source pages still release
  expert-by-expert.
- Per-tensor solver diagnostics print as it runs (`rel_err`, mean
  iterations, unconverged groups) — the same fp16-rounded-scale
  reconstruction error `MatrixStats` measures, so a bad tensor is visible
  immediately, not after a full pass. Expert stacks fold their per-expert
  stats into one line (mean + max `rel_err`) instead of printing one row
  per expert.

Validated on Qwen3-0.6B f16: K=2 full decoration (196 matrices → 392
planes, 840→217 MiB linears) loads through the qwen3 runners and generates
fluent text; a K=3 partial run (`--ptqtp-include blk.0.attn`) reproduces
the expected error ladder (rel_err ~0.068 vs dual's ~0.18) and the mixed
decorated/undecorated file serves correctly. MoE validated on
Qwen3-30B-A3B Q5_K_M: quantizing one layer's three 128-expert stacks at
K=2 (`--ptqtp-include blk.0.ffn_gate_exps,...`) peaks at a 105.8 MiB
working set, and the mixed file loads through the qwen3 MoE
pair-detection and generates correct text; per-expert plane bitwise
identity against standalone `quantizeMatrix` is pinned by
`ptqtp_gguf_tests`.

## Configuration guidance

- **Source precision**: quantize from bf16/f16 originals when available.
  The from-quantized path (e.g. a Q4_K_M file) works through the same code
  and degrades gracefully (~15–18% worse ppl), never collapses. f16 vs f32
  is immaterial (source rounding sits orders below the ternary error
  floor).
- **Plane count**: K=3 for accuracy (parity-class, see below); K=2 only
  with selective K=3 overrides on down_proj/o_proj — model sensitivity to
  the K=2 error grows sharply with model size (0.6B tolerates it at ×3.2
  ppl; 1.7B collapses ×17) and concentrates in the residual-writing
  projections, with a tail spread over q/k/v/gate/up.
- **lm_head** (`--head-planes`): quality-free at K=3 (measured Δppl ≈ 0),
  and with the x4 column-interleaved fused path (docs/TERNARY.md) the
  ternary head now also WINS on ARM: paired same-thermal-window decode runs
  on M1 Max (Qwen3-0.6B) measure the K=2 head ~7-8% faster end-to-end than
  the f16 head at both an f16 body and a K=2 body — the pre-x4 verdict
  ("keep the bf16 head on ARM; the ternary GEMV is ALU-bound") is
  overturned. Decorate the head for speed and memory alike (fully-ternary
  1.7B = 1.20 GiB of weights at baseline-parity ppl); x86-VNNI already
  favored it via the ~2× per-plane kernel margin.
- **Edge-layer skip**: subsumed by K=3 (buys ~nothing on top); useful as a
  cheap quality lever for K=2-budget deployments (first layers matter more
  than last).

## Measured (M1 Max, ReleaseFast; NLL = teacher-forced over 512 held-out tokens)

**Two-spirals MLP** (2→256→256→2 tanh, float-trained to 1.000, w2/w3
decorated post-training):

| variant | acc (exact-f32 path) | acc (deployed int8 path) | rel err w2 |
|---|---|---|---|
| float | 1.000 (CE 0.0042) | — | — |
| absmean b1.58, 1 plane | 0.670 | 0.670 | — |
| PTQTP K=1 | 0.644 | 0.644 | 0.474 |
| **PTQTP K=2** | **1.000** (CE 0.0127) | **1.000** (CE 0.0126) | **0.190** |

Single ternary planes collapse post-hoc; the dual decomposition holds full
accuracy on the deployed int8 path.

**Perplexity** (all data-free; decoration takes seconds at 0.6B, ~90 s at
1.7B K=3, multi-threaded):

| model / source | baseline | K=2 | K=2 +down3+o3 | K=3 |
|---|---|---|---|---|
| 0.6B f16 | 24.96 | 80.47 | 51.69 | **27.36** |
| 1.7B Q4_K_M | 19.41 | 330.96 | — | 21.71 |
| 1.7B BF16 | 18.57 | 184.45 | 45.70 | **18.43** |

K=3 from the bf16 original matches the full-precision baseline at 1.7B
(statistical parity; the greedy completion is flawless) and lands within
10% at 0.6B, with weight-space rel err 0.067 on the 27-level bound
(~1/3 of dual's 0.179). Solver diagnostics at these scales: mean ~17
iterations, unconverged groups ≤ 0.2%. 0.6B data-free edge-skip curve
(K=2 base): 1/1 → 69.0, 2/2 → 65.0, 3/3 → 51.6, 4/4 → 46.6.

**Decode speed** (1.7B vs its BF16 original, interleaved runs; fused
entry):

| config | t/s | vs baseline | ppl |
|---|---|---|---|
| bf16 baseline | 20.2–21.0 | 1× | 18.57 |
| K=2 | 47.1–48.3 | 2.3× | 184 |
| **K=3** | **39.5–39.8** | **1.9×** | **18.43** |
| K=3 + K=3 head (fully ternary) | 33.9 | 1.65× | 18.40 |
| K=2 +down3+o3 | 49.3 | 2.4× | 45.7 |

Against a Q4_K_M baseline (~44–48 t/s at 1.7B) K=2 is speed-parity and
K=3 ~0.75×; at 0.6B K=2 is ~1.9× the f16 source and parity with Q4_K_M.
Weights: 1.7B linears 3.2 GiB bf16 → 693 MiB (K=2) / 1040 MiB (K=3).
Kernel truth is `zig build bench-ternary`: per plane the TQ2_0 kernel is
~2.1× Q4_K on ARM and ~4.8× on x86-VNNI, so multi-plane configs win
outright on x86-class VNNI hardware. The ARM kernel is verified at ~86% of
the NEON ALU roofline (LLVM emits the fully-folded 10-ops-per-64-weights
sequence; hand asm has ≤14% headroom), and the ARM↔x86 per-instruction gap
is instruction-set density (sdot 16 weights/instr vs vpdpbusd 32; i8mm
smmla would close it on ARMv8.6+ targets).

## Limits and future work

- Measured dead ends, recorded so they are not re-chased: sparse exact
  outlier carry cannot rescue K=2 (carrying the top 1.56% |w| per group
  exactly improves rel err only 0.184→0.146 vs K=3's 0.069 — the K=2 gap
  is bulk 9-level resolution, not tails); per-128 packed scales (~1.3%
  fidelity, needs a block-layout change); E-core threads (even column
  splits go straggler-bound on heterogeneous cores); custom NEON asm
  (≤14% under the verified roofline).
- Decode dispatch: ~140 fork-joins/token remain at 1.7B (linears + attention
  + norms), worth ~3–5 ms against a ~64 t/s ARM ceiling at K=3. The
  dependency-respecting fix is a per-layer phase chain
  (`thread.parallelChained`, `exec/moe_chain.zig` precedent, family-local
  per the placement policy) — a second decode path with a parity burden;
  cheaper spin/dispatch-cost tuning should be measured first.
- Per-row selective third plane (plane3 over a chosen row subset + row
  index, selection data-free) would shape the K=2↔K=3 frontier finer than
  the per-projection overrides; unbuilt.
- The generic facade tq2_0 dot (non-PTQTP consumers) still pays one
  fork-join per call; the PTQTP path bypasses it via the fused entry.
- MoE expert stacks SERVE from persisted planes (resident and streamed,
  see Surfaces) but no in-tree walker decorates them yet — producing a
  K-plane expert GGUF takes an external converter running
  `ptqtp.quantizeMatrix` per expert and writing the `<name>.ptqtpK`
  siblings.
- Calibration-based extensions (activation-weighted solve, mean-correction
  bias, sample repair, self-calibration) live on the `feat/ptqtp-repair`
  branch, deliberately out of the data-free mainline.
- PTQTP-as-init for STE fine-tuning (`dotTernarySte`) is unexplored.

## Provenance

Method: arXiv:2509.16989v3 (PTQTP), reimplemented from Algorithm 1 /
Eq. 3–10; no reference code existed (the paper's repository link was empty
at porting time). Substrate: the TQ2_0 kernels, encoders, and block layout
of docs/TERNARY.md (ggml/llama.cpp lineage — MIT, see
docs/THIRD-PARTY-NOTICES.md).
