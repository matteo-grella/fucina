# Engram: conditional n-gram memory

Design record for `src/llm/engram.zig` — Fucina's port of DeepSeek's Engram
(arXiv 2601.07372; semantics reference `deepseek-ai/Engram`
`engram_demo_v1.py`, pinned in `tools/fetch_refs.sh`). API surface:
REFERENCE.md §13.11.

## What it is

Engram is a second sparsity axis beside MoE: **conditional memory**.
Where MoE spends a routed subset of FLOPs per token, Engram spends a
routed subset of *storage* — suffix n-grams of the token ids deterministically
address rows in large embedding tables, and the retrieved rows are gated
into the residual stream. The paper's measured claims (verified against
the primary source 2026-07-14): iso-parameter/iso-FLOPs gains over a pure
MoE baseline at 27B (MMLU +3.4, BBH +5.0, HumanEval +3.0), a U-shaped
sparsity-allocation law with the optimum near 20–25% of the sparse budget
in memory, and — the property that makes it a CPU-first technique — a
100B-parameter table served from host DRAM at ≤2.8% throughput cost,
because every address is a pure function of token ids known BEFORE the
layer executes.

That last property is the reason this module exists in Fucina: table rows
can live out-of-core (mmap/disk, the ExpertStore tier family) and be
prefetched with **zero speculation** while earlier layers compute. Nothing
else in the transformer has that: expert routing needs the hidden state,
attention needs the KV — Engram needs only the tokens.

## The mechanism (reference semantics, pinned)

1. **Token compression.** Raw ids map through a lookup table built by
   normalizing the tokenizer vocab (NFKC → lowercase → whitespace dedup;
   ids with identical normal forms collapse). The table is an *input* to
   the module (`HashPlan` `lookup`; identity when absent) — building it
   from a tokenizer is host tooling, not module logic. The pad id is
   compressed through the same table.
2. **Multipliers.** Per Engram layer, `max_ngram_size` odd int64
   multipliers drawn from `rng(seed + 10007·layer_id)` in
   `[1, 2·half_bound)`, `half_bound = (int64max / compressed_vocab) / 2`.
   The reference draws from numpy PCG64; Fucina draws natively (std PRNG)
   or accepts injected multipliers (`initWithMultipliers`) for bit-parity
   with reference artifacts. Multipliers persist in the state dict
   (frozen i64 entry), so a checkpoint is self-describing either way.
3. **Head table sizes.** For each layer, order, and head: consecutive
   distinct primes searched upward from `engram_vocab_size[order] − 1`,
   with the seen-set GLOBAL across the whole iteration — every head
   everywhere gets its own prime, which keeps the per-head hash functions
   independent.
4. **The hash.** For order `n` at position `t`:
   `mix = XOR_{k<n}(ids[t−k] · mult[k])` over WRAPPING int64 (out-of-range
   positions read the compressed pad id), and head `j`'s row is
   `mix mod prime[layer][n][j]` — floored mod (numpy `%`), so negative
   wrapped mixes stay in `[0, prime)`. Fucina implements this twice and
   pins them bit-equal: `hashInto` (host loop, the serving fast path) and
   `hashTensor` (integer tensor ops: wrapping `mul`, `bitXor`, `mod`,
   broadcast `add` — the ops added for this port, REFERENCE §4.19).
5. **Retrieval.** One concatenated `[Σ primes, head_dim]` table per
   layer; row indices carry per-head offsets, so retrieval is a single
   differentiable `gather` (scatter-add adjoint = the embedding
   backward), and the head rows flatten to `[seq, engram_hidden]`.
6. **Gating and injection.** Per hyper-connection stream `g`:
   `gate = sigmoid(signed_sqrt((rms(W_k·emb + b_k) · rms(h_g)) / sqrt(d)))`
   with `signed_sqrt(x) = sign(x)·sqrt(clamp_min(|x|, 1e-6))` and
   weighted RMSNorms (eps = `finfo(f32).eps`, the torch `nn.RMSNorm`
   default the reference relies on). Then
   `value_g = gate_g ⊙ (W_v·emb + b_v)`, and the module's output is
   `value + ShortConv(value)` where ShortConv = per-stream weighted
   RMSNorm (eps 1e-5) → **dilated causal depthwise conv1d**
   (`kernel_size` taps, dilation = `max_ngram_size`, no bias) → SiLU.
   The caller adds the residual. Plain residual-stream models are the
   `hc_mult = 1` case (`forwardResidual`); the reference's 4-stream
   hyper-connection layout is `hc_mult = 4`.

## What landed

- **Core ops added for the port** (all general-purpose, REFERENCE §4.4 /
  §4.16 / §4.19): integer `rem`/`mod` (pairing `divTrunc`/`divFloor`),
  bitwise `bitAnd`/`bitOr`/`bitXor`, one-sided `clampMin`/`clampMax`, and
  a `dilation` parameter through the whole `causalDepthwiseConv1d` chain
  (scalar + SIMD forward and both backwards, streaming-state validation
  at `dilation·(taps−1)` rows). Dilated depthwise is pinned by hand
  values and by a grouped-conv equivalence test including gradients.
- **`src/llm/engram.zig`**: `HashPlan` (geometry + hash, tensor-op and
  host paths), `Layer` (parameters + forward + `forwardResidual` +
  ShortConv streaming state), `Engram` (whole-model wrapper, registry,
  state dict). ~40 lines of integration surface for a model family:
  build a plan, build/load layers, call
  `hidden = engram_out + hidden` at the configured layers.
- **Graft mode** (`InitOptions.graft_zero_init`): value projection
  zero-initialized ⇒ module output is EXACTLY zero ⇒ adding Engram to a
  frozen pretrained model is bitwise identity at step 0, while gradients
  reach every parameter through the value path (gates and table learn as
  soon as `value_w` moves). This is the cheap-experiment mode: frozen
  base + LoRA + Engram tables, no pretraining required.
- **Parity**: `tools/gen_engram_goldens.py` (independent PyTorch/numpy
  implementation, torch 2.12) emits `src/llm/engram_golden_tests.zig`.
  The integer geometry — reference-drawn multipliers, the prime chain,
  compression, and every hash row — compares EXACTLY; forward output,
  loss, and the gradients of the hidden states and every parameter
  compare under the shared golden tolerance
  (`1e-5 + 2e-3·|expected|`).

## Gates (all green)

| Gate | Pin |
|---|---|
| Hash host path ≡ tensor-op path | bit-equal on wrap-forcing multipliers (`engram_tests.zig`) |
| Prime chain | exact values, global seen-set across layers/orders (`engram_tests.zig`) |
| Reference parity | multipliers/primes/rows EXACT + forward/backward goldens (`engram_golden_tests.zig`) |
| Graft identity | zero output at step 0, nonzero `value_w` gradient (`engram_tests.zig`) |
| Dilated depthwise conv | hand values + grouped-conv equivalence incl. gradients (`tensor_tests.zig`) |
| Integer ops | numpy int64 semantics: wrap, floored mod, xor (`tensor_tests.zig`) |
| Persistence | state-dict roundtrip incl. multipliers (`engram_tests.zig`) |

## Integration patterns

- **Pretraining / from-scratch** (nanochat-class): create
  `Engram.init(...)` beside the model, register its params on the same
  optimizer (`registerParams`), and at each layer in `layer_ids` add
  `try layer.forward(...)` output to the hidden states before attention.
  Hashing is per-sequence host work (`compressInto` + `hashInto`) —
  do it once per batch outside the graph.
- **Graft onto a frozen checkpoint**: same wiring with
  `graft_zero_init = true`; freeze the base (constants / frozen registry
  entries), train tables + projections (+ optional LoRA on the trunk so
  it learns to consume the injected signal). Step-0 outputs are bitwise
  identical to the ungrafted model — the free regression gate.
- **Serving**: `hashInto` per decoded token (a ring of the last
  `max_ngram−1` compressed ids per stream is all the state the hash
  needs); ShortConv streams through `conv_state` exactly like the other
  causal convs. Because addresses precede the forward, an out-of-core
  table tier can issue prefetches for layer L's rows while layer L−1
  computes — the ExpertStore composition (planned; the plan struct is
  deliberately tensor-free so a prefetcher can share it).

## Availability honesty

The upstream repo ships a mocked demo script only — no training code, no
checkpoints (DeepSeek V4 shipped WITHOUT Engram). There is no reference
artifact to load; parity is against the demo's mechanism, reproduced
independently in torch/numpy. Producing a useful Engram model in Fucina
therefore means training one (nanochat-scale from scratch, or the graft
mode above). Hash-collision behavior at small scale is studied in
arXiv 2601.16531; expect the iso-FLOPs gains of the 27B paper to need
re-measurement at desktop scale.

## Follow-ups

- Out-of-core table tier (mmap + deterministic prefetch through the
  ExpertStore seam) — the CPU headline; the hash plan is already
  shareable and tensor-free.
- Tokenizer-compression table builder (host tool over a GGUF tokenizer:
  NFKC/lowercase/whitespace normal forms).
- nanochat pretraining arm and the frozen-Qwen3 graft experiment
  (`graft_zero_init` + LoRA), with recall/task-exact probes.
- Batched/packed-segment forward for the trainers (the qwen3
  `packed_segments` decomposition applies — the hash is per-segment).
