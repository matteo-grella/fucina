<!-- docs-nav: group="Memory & compute" title="Subquadratic attention" weight=42 -->
# SCSA: Self-Calibrating Subquadratic Attention

A training-free replacement for the decode-time attention operator of a
pretrained causal Transformer. Weights, projections, and the KV cache stay
untouched; only the per-layer `softmax(QK^T)V` evaluation over the cached
keys and values is replaced by a sparse operator that reads a bounded
fraction of the context per step and self-calibrates its own thresholds
during the first decode steps.

Status: experimental, opt-in. Wired to the Qwen3 family over the f16 KV
cache: `models.research.subq` installs on the descriptor runner's
`Model.attention_override` seam and the stock `forwardStep` decodes
through it. Research tools:
`tools/bench_subq_decode.zig`, `tools/eval_subq_freerun.zig`,
`tools/bench_subq_kernels.zig`, `tools/bench_subq_scaling.zig`.

## 1. Contract

SCSA is a statistically contracted approximation, not an exact kernel:

- **Nothing is dropped.** Every cached key contributes to every output,
  either exactly (opened clusters, sinks, the recent suffix) or through a
  closed-form moment term inside the shared softmax normalization.
- **Quality is a calibrated per-head threshold**, expressed as an
  uncaptured attention-mass fraction `tau`, frozen by the operator itself
  from error curves it records while producing exact outputs during its
  calibration window. Typical greedy agreement with the dense operator is
  94-99% per step under the default tolerance; free-running generation
  tracks the dense model (retrieval prompts answered identically).
- **Speed grows with context.** The dense operator's per-step cost is
  linear in context; SCSA's exact reads stay a bounded fraction, so the
  crossover on this hardware sits near 8K context and the advantage widens
  from there. Below the crossover the dense operator should be used.

The contract is statistical, not certified: sound per-cluster bounds at
these dimensions force nearly every cluster open (measured), so the
calibrated threshold is the operating contract by necessity.

## 2. Mechanism

Per (layer, KV head), the operator maintains a **plan** over the sealed
cache prefix `[sink, seal_end)`:

- **Clusters.** Sealed keys are grouped by deterministic farthest-direction
  recursive splitting into clusters of at most `cluster_size` (128)
  members. Each cluster stores its member count `m`, key centroid `c_K`,
  value mean `c_V`, the top-`rank` (2) eigenpairs of the member-key
  covariance (power iteration with deflation), and the diagonal variance
  residual.
- **Packed copies.** Member keys and values are copied contiguously per
  cluster in the packed format (`q8_0` default: integer-dot scoring with
  the query quantized once per head call; `f16` alternative), so an opened
  cluster is one streaming read.
- **Incremental maintenance.** Every `rebuild_interval` (512) appended
  tokens only the growing tail region is re-split and re-packed; once a
  region reaches `block_size` (4096) rows it freezes permanently, so
  maintenance cost is bounded by the block size, not the context length.
  Sinks (`sink` = 4) and the unsealed suffix (`recent` = 32 plus the
  region not yet rebuilt) are always attended exactly.

Per decode query, per head:

1. **Priorities.** Every cluster receives the second-cumulant log-mass
   estimate `a + log m + beta^2 q^T Sigma q / 2`, with `a = beta q . c_K`
   and `Sigma` the rank-2-plus-diagonal covariance summary; for all query
   heads of a layer these are three matmuls against the layer's
   concatenated summaries.
2. **Estimate-only stop.** Clusters are opened best-first until the
   estimated uncaptured-mass fraction falls below the head's `tau`. The
   opened set is decided entirely from the estimates, before any read, so
   the reads run as one ascending sweep in which adjacent opened clusters
   coalesce into single streaming batches (online-softmax gauge, catalog
   row-block kernels).
3. **Moment tail.** Every unopened cluster contributes `m exp(a)` at its
   value mean inside the shared normalizer.
4. **Output** = (exact numerator + tail numerator) / (exact mass + tail
   mass).

**Self-calibration.** For the first N decode steps (32 in the tools) the
operator opens every cluster (exact outputs) while counterfactually
replaying its stop rule at a grid of thresholds and recording, per head,
the relative output error each threshold would have produced. At the end
of the window each head freezes the largest threshold whose median error
meets the tolerance (default 0.025). Thresholds are mass fractions, so
they transfer across context lengths; the same taus serve 8K and 32K.

**Hierarchical mode** (`Config.hierarchical`). Selection walks a balanced
interval tree over the leaf clusters (internal nodes carry exactly
aggregated count / centroid / value mean / full diagonal variance),
expanding the highest-estimated-mass frontier node until the stop holds;
unexpanded nodes contribute their aggregated tail term at their own
granularity, and calibration replays the same walk. Selection cost then
tracks the opened set instead of the cluster count: parity with the flat
mode at 32K, constant selection cost measured from 32K to 524K rows. It
is the long-context mode; the flat mode is the default at native contexts.

## 3. Usage

```zig
const models = @import("fucina_models");
var sq = try models.research.subq.State.init(allocator, num_layers, q_heads, kv_heads, head_dim, .{});
defer sq.deinit();
model.attention_override = models.research.subq.attentionOverride(&sq);
defer model.attention_override = null;
try sq.startCalibration(0.025);         // first N decode steps run exact
// ... N steps of model.forwardStep(&ctx, &kv, &one_token, pos)
sq.finishCalibration();                 // per-head taus frozen
// ... continue decoding through the installed override
```

`Config` fields: `cluster_size` (128), `rebuild_interval` (512),
`block_size` (4096), `sink` (4), `recent` (32), `tau_default` (0.05, used
before calibration or when a head records no rows), `packed_format`
(`.q8_0` / `.f16`), `rank` (2), `power_iterations` (4), `hierarchical`
(false). `State.taus` may also be loaded from JSON (`loadTausJson`,
`{"layer:head": tau}`) or set directly. `State.serial` forces the per-head
work serial (diagnostics). Counters: `rebuild_count`, `stat_opened_rows`,
`stat_exact_rows`, `stat_scored`, `stat_calls`.

Requirements checked at init: `q_heads % kv_heads == 0`, `kv_heads <= 64`,
`q_heads <= 128`, `head_dim` a multiple of 8 and at most 512, `rank <= 8`.
The KV cache must be f16 (the operator reads it directly). A cache
truncated below a plan's sealed region drops that plan and replans.

## 4. Measurements

Protocol: Qwen3 GGUF models, f16 KV cache, greedy decode, held-out
natural-text prompts, 96 timed steps after a 32-step calibration at
tolerance 0.025, per-step greedy agreement against the dense operator fed
the same token stream (teacher-forced), both arms sequential in one process
on an idle Apple-silicon machine, rows measured coldest-first in one
battery (`bench_subq_decode`, log `final_battery_cold2.log`).

| Model | Context | dense tok/s | SCSA tok/s | speedup | agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.6B Q8_0 | 1,024 | 112.5 | 81.1 | 0.72x | 97.9% |
| 0.6B Q8_0 | 8,192 | 44.5 | 46.1 | 1.04x | 93.8% |
| 0.6B Q8_0 | 16,384 | 26.3 | 35.7 | 1.36x | 96.9% |
| 0.6B Q8_0 | 32,768 | 13.4 | 26.1 | 1.95x | 96.9% |
| 0.6B Q8_0, sustained 992 steps at 32K | | 14.1 | 25.7 | 1.82x | 98.9% |
| 1.7B Q4_K_M | 8,192 | 24.9 | 27.5 | 1.10x | 97.9% |
| 1.7B Q4_K_M | 16,384 | 15.0 | 21.1 | 1.41x | 100% |
| 1.7B Q4_K_M | 32,768 | 8.5* | 20.7 | 1.67-2.4x* | 90.6% |

The sustained row crosses a plan-maintenance boundary inside the timed
window (maintenance is measured, not excluded). (*) The final 1.7B/32K row's
dense arm ran thermally degraded (a cool dense measurement of 12.4 tok/s
gives 1.67x); its 90.6% agreement is the reproducible per-segment
calibration variance of 1.7B at 32K.

Packed format (same battery, 0.6B): q8_0 vs f16 at 32K reads 26.1 vs 21.9
tok/s at equal-or-better agreement. Tolerance is the speed knob (0.6B, 8K,
f16: 0.025 gives 0.91x at 97.9%, 0.035 gives 0.96x at 95.8%, 0.05 gives
0.98x at 93.8%; at 32K, tolerance 0.1 gives 2.37x at 84.4%).

Free-running behavior (`eval_subq_freerun`, measured on the f16-packed
flat-rebuild predecessor of this kernel; the mass model and calibration are
unchanged, the re-run on the current kernel is pending): both arms feeding
their own greedy tokens after a shared calibration window, dense-judged
mean NLL delta of the sparse arm's continuations `+0.008` nats over eight
runs (0.6B and 1.7B, 8K-32K); free-running retrieval 19/19 (passkey 9/9,
multi-needle 6/6, variable tracking 4/4) identical to dense.

Selection scaling (`bench_subq_scaling`, synthetic clustered KV, one layer,
selection-only at tau = 1): flat 0.46 / 1.32 / 2.95 ms at 32K / 131K /
524K rows; hierarchical 0.056 / 0.032 / 0.046 ms.

## 5. Limits and scope

- Experimental and opt-in; Qwen3 family, f16 KV cache, single-stream
  decode. Prefill is unaccelerated and the initial plan build after a long
  prefill is a one-time cost (13-17 s at 32K on 0.6B).
- Below roughly 8K context the dense operator wins.
- Agreement is calibrated, not exact; per-segment calibration variance is
  visible on 1.7B at 32K.
- Speed ratios are hardware- and thermal-state-dependent; the table above
  is one machine, measured cold, sequential arms.
- Contexts beyond the model's native window are untested.
