# Speculative Decoding — Design Record

This document records the design of Fucina's **training-free, draft-model-free
speculative decoding**, the losslessness contract and its proof obligations,
the verify economics that drive every policy decision, the bench results with
their honest caveats, and the **adjudications** — the approaches from the
literature that were evaluated and rejected, with the conditions that would
reopen them. Grounded in the source; file:line anchors are included so claims
can be re-verified rather than trusted.

The short version: drafts come from cheap deterministic indexes over text the
model has already seen (the conversation itself, injected reference documents,
recycled verification logits) — **no draft model, no training, no extra
weights**: a ~4.6 MiB recycling matrix plus a suffix-automaton (SAM) index
that grows at ~110 B/token. One batched forward verifies the draft;
the committed stream is provably the same one plain decoding would produce,
for greedy *and* sampled decoding. A cost-aware gate makes the whole feature
**never-a-loss**: tasks it can't accelerate run at 0.98–0.99x (probe
overhead), tasks it can run at 1.1–2.3x. Scope: `qwen3` and `gemma4` (any
model with the duck-typed `forwardStep`/`forwardStepAllLogits`/`KvCache`
contract). `qwen35` (Qwen3.5's hybrid Gated-DeltaNet architecture) is
structurally out of scope — see §11.

---

## 1. Architecture

Three layers, each independently testable:

```
DraftSource (vtable)          src/llm/speculative/core.zig:74
  ↑ implemented by
SpeculationIndex (cascade)    src/llm/speculative/cascade.zig:130
  conversation SAM ── frozen reference SAMs ── Token-Recycling matrix
  src/llm/speculative/sam_index.zig:103        src/llm/speculative/recycling.zig:144
  ↓ drafts verified by
SpeculativeDecoder(Model)     src/llm/speculative/core.zig:476
  one batched forwardStepAllLogits + full sampler pipeline per row
  KvCache.truncate drops rejected rows        src/llm/kv_cache.zig:310
```

- **`DraftSource`** is a three-method vtable: `suggest(context, buf)`
  proposes a continuation of the committed stream, `observe(committed)` feeds
  index updates, optional `observeTopK(rows)` receives top-K candidates from
  the verification logits (skipped — including the top-K compute — when the
  source doesn't want it). Externally injectable: the decoder works with any
  deterministic proposer.
- **`SpeculativeDecoder(Model)`** runs the decode loop step: ask the source
  for a draft, run **one** batched forward over `[carried token, draft...]`
  via `forwardStepAllLogits` (qwen3/model.zig:320, gemma/gemma4.zig:652 — same as
  `forwardStep` but no `last_query_only` narrowing, returns `[k, vocab]`),
  sample every row with the full pipeline, commit the longest prefix the
  target model itself would have produced, truncate the KV cache back to the
  accepted length. Drafts shorter than `min_draft` (2) fall back to a plain
  step.
- **New model-side primitives** (the only model code speculation needed):
  `KvCache.truncate` — a len-clamp; per-(position, kv-head) row layout makes
  rewind trivial for both f16 and q8_0, verified bitwise against a fresh
  cache by the truncate + re-append test (kv_cache_tests.zig:418) — and
  `forwardStepAllLogits`.

## 2. The losslessness contract

**Claim: the speculative decoder commits exactly the token stream plain
decoding would commit — greedy and sampled — through ONE code path.** Because
every draft source is deterministic (a one-hot proposal distribution q),
speculative rejection sampling (Leviathan et al., 2023) degenerates to: run
the full sampling pipeline (penalties, temperature, top-k/top-p/min-p) on the
target logits at each verified position; accept while `sampled == draft[i]`;
at the first mismatch the sampled token *is* the correction token (provably
target-distributed); on full acceptance the (k+1)-th row's sample is a free
bonus token. Greedy is the same path — temperature ≤ 0 makes the sampler an
argmax. Token IDs are compared, never probabilities.
(speculative/core.zig:1–46 is the normative header.)

The contract decomposes into proof obligations, each with a test:

1. **One RNG draw per committed token.** Positions past the first mismatch
   are never sampled, so the RNG stream advances exactly as in a plain run.
   Tested by the sampled replay-equivalence test (temperature 0.8, top-k 20):
   token streams identical for the perfect *and* the garbage source.
2. **Replay equivalence of sampler inputs.** A `VerifyRowHook` captures every
   pre-penalty logits row; at every commit position the speculative run's row
   must equal the plain run's row for the same committed prefix — **bitwise on
   `-Dbackend=scalar` and `-Dblas=none`** (`batch_rows_bitwise`,
   speculative/core_tests.zig:269), tolerance 1e-4 under vendor BLAS, whose m-dependent
   GEMM kernels reassociate (~1e-6 rel measured drift). "Lossless" therefore
   means precisely: *same distribution always; same sample stream whenever the
   logits match bitwise* — which they do below the m-dependent kernel
   thresholds (fused K-quant FFN seq ≥ 12, tiled attention seq ≥ 48) on
   BLAS-free builds.
3. **Penalties condition on the hypothetical prefix.** Accepted tokens enter
   `history` before the next row is sampled, so frequency/presence/repeat
   penalties see exactly the tokens their logits row is conditioned on;
   rejected drafts never touch `history` (no rollback needed). The test is
   discriminating, not vacuous: re-running the sampler on captured rows with
   *stale* (iteration-start) history provably changes at least one pick
   (speculative/core_tests.zig:520-572).
4. **Adversarial sources can't corrupt the stream.** Perfect / garbage /
   alternating sources all produce token-for-token plain output (greedy
   losslessness test); the cache never retains unverified rows, including on
   error unwind (`errdefer kv.truncate`, speculative/core.zig:616/:656).
5. **Gating is orthogonal.** The gate decides *when* speculation runs, never
   *what* is committed (end-to-end gate test asserts identical streams while
   the gate trips, backs off, and re-probes).

## 3. Verify economics and break-even math

Measured with the `--spec-bench` probe mode (examples/qwen3/main.zig): cost of one
verify-k forward in plain-step equivalents, best-of reps, M1 Max ReleaseFast.

**Dense Qwen3-0.6B-Q4_K_S** (the shipped `default_cost_table`,
speculative/core.zig:220):

| draft k | verify cost (plain steps) | conservative break-even acc ≈ cost/k |
| --- | --- | --- |
| 2 | 1.65 | ~83% |
| 4 | 1.42 | ~36% |
| 8 | 2.84 | ~36% |
| 16 | 4.5 | **~28%** |

The table is non-monotonic at the low end (verify-4 cheaper than verify-2 —
small-m kernel dispatch effects), which is why it is a measured table with
interpolation, not a formula.

**Mixture-of-experts (MoE) Qwen3-30B-A3B: verify-4 = 3.7 plain steps.**
Batching defeats the MoE decode advantage: each verify row activates its own
experts, so weight reuse across rows is minimal and a verify-m forward costs
nearly m plain steps. Conservative break-even ≈ 3.7/4 ≈ **93% acceptance** —
out of reach for retrieval sources on general text, so **the gate carries
MoE**: speculation auto-disables there and costs only the probe overhead.

The exact math: a verify over k draft tokens costs C(k) plain-step
equivalents and commits a+1 tokens (a = accepted prefix; the +1 — correction
*or* bonus — is always free). It beats plain decoding iff `a + 1 > C(k)`, so
strict per-verify break-even is `a = C(k) − 1` (dense k=16: ~22% acc; MoE
k=4: ~68%). The quoted planning figures use the conservative `acc ≈ C(k)/k`
form that ignores the free token; the gap between the two bounds is the
margin for overheads the table doesn't price (probe steps, fallback churn,
and short low-k drafts whose per-token economics are worse).

## 4. CostGate — why hybrid static + EWMA

`CostGate` (speculative/core.zig:266) gates on **estimated speedup**, not tokens
per step:

```
est_speedup = committed_tokens / verify_cost_in_plain_step_equivalents
```

over a rolling window of verify steps (fallback steps cost the same as plain
steps and never enter the window — only verify economics can lose time). The
predecessor flat tokens/step gate kept e.g. 1.56 tok/step at k≈8 alive as a
"win" while actually losing ~45% (verify-8 ≈ 2.84 plain steps).

**Why the cost model is hybrid.** The static measured table gives the cost
*curve's shape*; live verify/plain timings (when the decoder has a clock,
`decoder.io`) continuously **rescale** it through a clamped
exponentially-weighted moving average (EWMA) multiplier (`scale_alpha` 0.2,
clamp [0.25, 4.0], a verify never costs < 1 plain step).
A **pure-live model proved non-robust in practice: one noisy timing sample
(scheduler stall, thermal hiccup) tripped a winning run off.** Under the
hybrid, one sample moves the estimate by at most 20% of its clamped
deviation — it cannot flip the gate — while a model whose true economics
differ from the table (a different size/quant/machine) is learned within a
few verifies, and the table alone applies when no clock is set
(`verifyCost`, speculative/core.zig:365-375; cost-table/gate tests at
core_tests.zig:586/:602).

Policy constants (all in `Options`, speculative/core.zig:157):

- **Hysteresis 1.0 / 1.1**: speculate while est_speedup ≥ `min_speedup`
  (1.0); a re-probe re-enables only at ≥ 1.0 + `probe_margin` (0.10), so a
  marginal regime doesn't flap.
- **Exponential backoff 128 → 1024**: `reprobe_after` 128 disabled steps
  before the first 4-verify probe (`probe_steps`), doubling per failed probe,
  capped at `reprobe_max` 1024; a passed probe restarts at the base.
- **Acceptance-adaptive budget**: `budget = max(min_draft, 2 +
  ceil(acceptance · max_draft))` over the window — low-acceptance phases
  verify small cheap drafts instead of max_draft losses.
- Evaluation is per-verify once the window holds ≥ `min_window_drafted` (8)
  drafted tokens and ≥ 2 verifies (one verify is too noisy to act on): clear
  losses trip fast.

Measured effect (M1 Max, Qwen3-0.6B Q4_K_S): free-form 0.83x → **0.99x**,
reference-injected 0.86x → **0.98x** — speculation is now **never-a-loss**;
the residual 1–2% is probe overhead.

## 5. The SAM index

`src/llm/speculative/sam_index.zig` — an **online suffix automaton** (SAM) over
token streams (SAM-Decoding / SuffixDecoding lineage). Why a SAM and not
n-gram hashes: O(1) amortized online extension, an **exact, unbounded**
longest-suffix-match length (which drives the adaptive draft budget), and the
same construction doubles as a frozen index over reference documents.

- **Self-match exclusion** (the subtle part, speculative/sam_index.zig:22–75): the trivial
  SAM matches the stream against itself. Drafting needs the longest suffix
  with an occurrence ending *strictly before* the current position. Two
  mechanisms deliver this exactly: **cursor-before-extend** (the match cursor
  advances with t_i against the automaton of t_0..t_{i−1}, then the automaton
  is extended — a successful transition proves a prior occurrence; clone
  fix-up redirects the cursor when extension splits its state) and the
  **sample discipline** (each state stores one occurrence end index, every
  write provably a prior endpos member; clone samples are inherited, not set
  to the current end; recency refreshes are deferred one step).
- **Recency drafts**: drafts follow the *most recent* prior occurrence seen
  by the construction walk / cursor path (recency wins for code and editing
  flows, where the latest revision of a passage is the one worth copying).
- **Frozen Cursor mode (the retrieval-injection seam)**: build a SamIndex per
  tokenized reference document, `freeze()` it, then run any number of
  external `Cursor`s over it per conversation. Freezing is a correctness
  requirement, not an optimization: appends clone states and only the
  internal cursor gets the fix-up — external cursors over a growing automaton
  would dangle.
- **Verification**: `matchLen` and drafts are property-tested against brute
  force after every append (random + adversarial streams, alphabets 2/5/31);
  the classical bounds states ≤ 2n+1, transitions ≤ 3n are asserted. A failed
  append poisons the index (degraded mode: queries return 0 forever) instead
  of serving desynced drafts.
- **Memory: ~110 B/token** (16 B/state × ≤2n, one global transition hash map
  ≤3n entries, 8 B edge-pool mirror, 4 B/token stream copy).

## 6. The Token-Recycling matrix

`src/llm/speculative/recycling.zig` — the fallback when no index matches (Token Recycling,
Luo et al. 2024). Verification already computes full logits per position and
throws away everything but the sampled token; recycle them instead: one row
per vocab token holds the most recent top-K (K=8) next-token candidates
observed via `observeTopK`, plus committed-bigram move-to-front. Drafting is a
top-1 chain walk from the last committed token. Memory: vocab × 8 u32 =
**4.64 MiB** at the Qwen3 151,936 vocab; cold start is a sentinel fill (no
seen-bitmap).

## 7. The cascade policy

`SpeculationIndex` (speculative/cascade.zig:130) composes the sources behind one
`asDraftSource()`:

1. **conversation** — online SAM over every committed token (prompt +
   generated), auto-fed through `observe`;
2. **references** — frozen SAMs injected via `addReference`, each with a
   per-conversation cursor;
3. **recycling** — the fallback chain (cap 8).

Selection: query every source's match length, **draft from the longest
match** (ties → conversation: recency in the live stream beats a static
document; among references, first-added). Draft budget grows with match
confidence: `budget = 2 · (1 + match_len)` (β=2). Matches shorter than
`min_match` (2) fall through to recycling; an unseen recycling row drafts 0
and the decoder does a plain step.

**Per-source muting**: each source keeps a rolling acceptance window over its
last **64** drafted tokens; when a filled window drops below **20%**
acceptance the source is muted for **128** committed tokens, then re-probed
with a fresh window. No source dies permanently, and muted sources keep
observing (SAM appends, cursor advances, recycling updates) so they are in
sync at the re-probe. This is the source-*selection* gate; the decoder's
CostGate owns the global verify economics. Acceptance is inferred at the
`suggest` → `observe` seam (longest common prefix of draft and next committed
slice) — a heuristic for selection only, losslessness is unaffected.

## 8. RAG injection — usage and the tokenize-externally contract

The reference seam — retrieval-augmented generation (RAG) style injection of
documents the model is likely to quote — is deliberately token-based:
`addReference(tokens: []const usize)` (speculative/cascade.zig:211). **The
caller tokenizes** — the index never sees text. The contract that makes it
work: reference documents must be encoded with the *same tokenizer the model
decodes with*, or the SAM never aligns with the committed stream and
acceptance is zero. This is not theoretical — it is exactly how a
since-fixed pretokenizer bug (625 tokens vs llama.cpp's 500 on a code
fixture) silently broke the code-edit task (§10). `--tokenize` (the oracle
mode, §9) exists to audit this parity.

`addReference` catches the new cursor up over the already-observed committed
stream, so a document injected mid-conversation matches existing context
immediately; `clearReferences` drops all documents (and any pending
accounting). At the chat level: `Conversation.addSpecReference(tokens)`
(chat.zig:255, requires `Options.speculation`).

## 9. CLI and chat usage

Model weights are not shipped with the repository: download a Qwen3 GGUF
first (e.g. the Q4_K_S quantization of Qwen3-0.6B from Hugging Face) and
place it under `models/`, or pass any path to a GGUF you already have.

```sh
# Lossless speculative generation + acceptance stats:
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  --prompt "..." --gen 128 --spec

# Inject reference documents (the RAG seam; repeatable, up to 8):
zig build qwen3 -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q4_K_S.gguf \
  --prompt "..." --gen 128 --spec --spec-ref doc.txt

# Verify-economics probe (the cost-table source):
zig build qwen3 -Doptimize=ReleaseFast -- <model.gguf> <ids> --spec-bench

# Tokenizer parity oracle: one token id per line, no model weights needed
# (compare against `llama-tokenize --ids --no-escape`):
zig build qwen3 -- <model.gguf> --tokenize file.txt
```

Reported stats: `spec: steps=... committed=... (N tok/step) ... (P% acc)` from
the decoder plus the per-source summary
(`spec sources: conversation a/d ref0 a/d recycling a/d [muted]`).

Library use: `chat.Options{ .speculation = true, .spec_options = .{...},
.io = io }` (chat.zig:132) — the `Conversation` owns a `SpeculationIndex` +
decoder; a `TurnGate` (chat.zig:570) keeps the conversation SAM byte-exact
across trimmed turns by filtering `observe`/`observeTopK` so the index never
learns tokens the turn trim discards (stop marker + overshoot). Setting `.io`
enables the live cost rescale; without it the gate runs on the static table.

## 10. Bench results (and the honest caveats)

M1 Max, ReleaseFast, Qwen3-0.6B-Q4_K_S, greedy unless noted. Speedup =
speculative vs plain decode tok/s on the same prompt, serial runs.

| Task | Speedup | Acceptance / notes |
| --- | --- | --- |
| Grounded copy (pre-tokenizer-fix prompt encoding) | **1.47x** | 70% acc, 6.6 tok/step |
| Same prompt, post-fix re-encode | 1.04x | 41% acc — see caveat below |
| Verbatim-repetition microcase | **2.3x** | 100% acc, ~11 tok/step |
| Code edit (post-tokenizer-fix) | **1.12x** | 84.6% acc (was 0.79x pre-fix) |
| Free-form generation | 0.99x | gate auto-off; was 0.83x pre-gate |
| RAG-injected reference | 0.98x | ref source 33/68 accepted — injectable index validated; was 0.86x |
| Grounded copy, sampled t=0.7 (pre-fix prompt) | **1.20x** | one RNG draw per committed token holds |

Context (not a controlled A/B): llama.cpp's `llama-lookup` reports 93%
acceptance with max-draft 3 on the same grounded text — a shorter-draft,
higher-acceptance operating point than Fucina's budget policy picks.

Caveats, recorded deliberately:

- **Task dependence is the headline.** Retrieval-shaped workloads (copy,
  edit, repetition, agentic re-statement) win 1.1–2.3x; free-form generation
  has nothing to retrieve and the honest number is ~1x *because the gate
  turns speculation off* — the 0.98–0.99x floor is the price of probing, and
  "never-a-loss" is a gate property, not a drafting property.
- **The tokenizer-fix / grounded-task story.** The 1.47x and the 1.04x rows
  are the *same prompt text*. Fixing the qwen2 pretokenizer (to token-ID
  parity with llama.cpp) changed the prompt's encoding; the greedy generation
  itself then diverged from the reference wording (acceptance 70% → 41%), and
  the speedup fell with it. The gate was verified non-limiting on the
  post-fix run (a forced-open run is identical). Lesson kept on record:
  acceptance — and therefore speedup — is a property of (model × exact token
  stream), and single-prompt speedups are fragile; the durable claims are the
  economics table, the gate floor, and the acceptance-conditional wins.

## 11. Adjudications

Recorded so these decisions are not silently re-litigated. Each candidate was
prototyped or paper-audited against the measured verify economics (§3).

- **Jacobi / plain parallel decoding — NO SLOT.** Free-form 1.05x in our
  setting: without retrieval structure, fixed-point iteration on CPU pays the
  batched-forward premium for ~1 extra token per step. Strictly dominated by
  the cascade + gate on every measured workload.
- **Lookahead decoding — NO SLOT.** 1.13x measured independently at **50–120x
  the FLOPs** of greedy decode (n-gram pool generation + verification
  branches). On GPUs that compute is idle and free; on CPU it is the
  scarce resource — the wrong trade by two orders of magnitude. It also
  effectively requires tree-mask attention to amortize. **Revisit iff** (a) a
  tree-mask attention kernel lands (§ tree verification below) *and* (b) a
  real workload shows high per-step uncertainty with measured spare compute
  (e.g. small model on a many-core x86 server where decode is
  bandwidth-bound).
- **Layer-skip self-speculation (SWIFT / CLaSp) — DEFERRED.** Published gains
  concentrate at 7B+; unproven at 0.6–4B where Fucina decode lives, layer
  skipping interacts with the resident-quantized weight layout, and Token
  Recycling already reaches the same acceptance band for ~5 MiB and zero
  model surgery. Re-evaluate when a ≥7B dense model becomes a primary target.
- **Tree verification (multi-candidate drafts) — DEFERRED.**
  Linear drafts capture the retrieval gains that exist on CPU today
  (§10), and the verify-cost table says wide trees multiply exactly the cost
  that is already the constraint. The natural next step is **batched
  candidate verification**: verify the top-2/3 cascade candidates as
  independent rows (no tree mask needed — rows share the prefix via the KV
  cache), then re-adjudicate masked trees with real numbers.
  `Recycling.topkOf` already exposes the row needed for tree-style drafting.
- **qwen35 — OUT OF SCOPE, structural.** Gated-DeltaNet's recurrent state
  cannot rewind: rejecting a draft would require restoring conv/delta-scan
  state at an arbitrary earlier position, which the recurrence does not
  support (and checkpoint-per-position would cost more than verification
  saves). The decoder is deliberately duck-typed on the rewindable-`KvCache`
  contract (speculative/core.zig:473-475); qwen35 does not satisfy it.

## Addendum 2026-07-02 — gemma4 chat composition + loud gate validation

- **Speculation composes with gemma4 chat.** `chat.Conversation(Model, Tok)` is
  genuinely generic (no family import in `chat.zig`), and `examples/gemma4/main.zig`'s
  `--chat`/`--repl` run on it with `--spec` — verified byte-identical greedy output
  with and without `--spec` on the real 26B GGUF. The §1 scope line ("qwen3 and
  gemma4") is exercised end-to-end from both CLIs.
- **`CostGate.init` validates its options loudly** instead of accepting degenerate
  configs: `RateWindowTooSmall`, `ProbeStepsZero`, `CostTableEmpty`, `ReprobeAfterZero`
  (speculative/core.zig:318-321; every-build-mode test at core_tests.zig:878).
- **Stop handling vs the one-draw contract:** `chat.Options` gained `extra_stop_ids`
  (verify-safe: token-ID comparison) and `stop_sequences` (text-level); combining
  `stop_sequences` with `speculation` is an init error (chat.zig:197) because the
  sequence-completing token could be accepted mid-verify-batch, breaking the
  one-RNG-draw-per-plain-committed-token contract (§2 obligation 1).
  `SpeculationIndex.truncatePending` (speculative/cascade.zig:403) keeps the
  per-source acceptance accounting aligned when the `TurnGate` truncates at a stop.
