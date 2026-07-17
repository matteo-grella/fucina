# Constrained Decoding — Design Record

This document records the design of Fucina's **grammar/JSON-schema
constrained decoding**: the pluggable logit-processor seam on the shared
sampler, the vendored [llguidance](https://github.com/guidance-ai/llguidance)
engine behind it, the argument for why the seam composes with speculative
decoding *without a rollback primitive*, the grammar-driven drafting layer
that turns a constraint from a speculation-killer into a speculation
accelerator, and the **adjudications** — designs evaluated and rejected, with
the conditions that would reopen them. Grounded in the source; file:line
anchors are included so claims can be re-verified rather than trusted.

The short version: every decode path in the tree samples through one
`Sampler.next`, so a single optional hook there — mask the logits row before
the pipeline, observe the selected token after — gives every model family
constrained decoding with **zero decode-loop changes**. The mask comes from
llguidance (compiled JSON schema / regex / Lark grammar → per-step token
bitmask, ~tens of µs of pure CPU work per token), vendored under
`vendor/llguidance/` and built only under `-Dllguidance=true`; the seam
itself is pure Zig and always available. Constrained output is
token-for-token identical across the plain, speculative, and lockstep-batch
paths — proven greedy and sampled — and with the grammar-drafting layer the
constraint *raises* speculative acceptance instead of muting it (measured:
0% → 83% on JSON-schema chat, output byte-identical).

Scope: any runner built on `llm.sampler.Sampler` — today the qwen3 and
gemma4 CLIs (`--json-schema JSON|@FILE`, `--lark GRAMMAR|@FILE`,
`--regex PATTERN`) and anything embedding `chat.Conversation`
(`Options.logit_processor`). Non-autoregressive paths (diffusion_gemma's
entropy-bound sampler, omnivoice's MaskGIT) are structurally out of scope —
they do not sample token-by-token through the shared sampler.

---

## 1. Architecture

Two layers with one optional third, each independently testable:

```
LogitProcessor (vtable)          src/llm/logit_processor.zig:35
  process(logits, history)         mask/bias the row before sampling
  commit(token)                    observe every selected token
  reset()                          re-arm per assistant turn
  forcedTokens / validPrefixLen    optional structural lookaheads
  ↑ hosted by
Sampler.processor                src/llm/sampler.zig:58
  process at next() entry :73, commit on every exit path :104
  ↑ installed via
chat.Options.logit_processor     src/llm/chat.zig:159
  per-turn reset :304, sendBatch share guard :412
  ↑ implemented by (opt-in, -Dllguidance=true)
llguidance Constraint            src/llm/llguidance.zig
  grammar compile + token bitmask over a Fucina-tokenizer bridge
  ↑ drafted by (when speculation is on)
ConstrainedSource                src/llm/speculative/constrained.zig:36
  forced spans → certain drafts; invalid drafts pruned pre-verify
```

The layering is strict: `logit_processor.zig` knows nothing about grammars,
`llguidance.zig` knows nothing about chat or speculation, and
`constrained.zig` knows only the two vtables it bridges (`LogitProcessor` ↔
`DraftSource`). A bias list, a banned-token rule, or a watermarking scheme
plugs into the same seam with no llguidance involvement — and conversely,
the llguidance build flag being off removes zero functionality from the
seam itself.

## 2. The seam — why inside the Sampler

The decision with the most leverage in this design is *where* the processor
hook lives. Every decode path already funnels through
`Sampler.next(ctx, logits, history)`:

- `chat.Conversation.send` (plain turn decode),
- `chat.Conversation.sendBatch` (lockstep multi-stream `sampleStep`),
- the speculative decoder's plain step *and* every verify row
  (`speculative/core.zig`),
- every hand-rolled runner loop (`examples/qwen3/main.zig` completion/bench,
  multi-stream arms).

Hooking the sampler therefore means: **implement once, constrained
everywhere** — including paths that did not exist when the constraint was
written, as long as they sample through a `Sampler`. The alternative
(wiring a mask call into each loop) was rejected; see §8.

The contract (`src/llm/logit_processor.zig:35`):

- `process(logits, history)` mutates one `[vocab]` f32 row in place before
  penalties/temperature/top-k/top-p/min-p run. A grammar mask writes `-inf`
  over forbidden ids; `-inf` survives the penalty pass unchanged (divide,
  multiply, and subtract all map `-inf → -inf`), so ordering against
  penalties is not semantically load-bearing.
- `commit(token)` observes the selected token, **exactly once per `next`
  call, on every exit path** — greedy included
  (`src/llm/sampler.zig:104,146`). This is the property everything else
  leans on (§4).
- `reset()` re-arms the state machine for a fresh constrained region.
  `chat.Conversation` calls it in `beginTurnTokens`
  (`src/llm/chat.zig:304`) — the shared turn prologue of `send`, `sendSpec`
  and `sendBatch` — so one constraint instance governs each assistant reply
  independently across a multi-turn conversation.
- A processor that leaves no selectable candidate is
  `error.AllTokensMasked` (`src/llm/sampler.zig:103,120`): a broken
  constraint fails loudly instead of silently sampling from a masked-out
  distribution. llguidance never produces an empty mask on a healthy
  matcher (a terminal grammar forces the stop token instead), so in
  practice this fires only on genuinely broken custom processors.
- One processor per decode stream, single-threaded, adjacent to its
  sampler. `sendBatch` enforces this: two streams sharing one processor
  pointer is `error.SharedBatchProcessor` (`src/llm/chat.zig:412`).

Stop handling needs **no new mechanism**: when a grammar completes, the
mask allows only the configured stop/EOS token, the sampler can only select
it, and the existing stop-token checks end the turn. Termination composes
with `extra_stop_ids`, `stop_sequences`, and the response budget unchanged.

## 3. The engine — vendored llguidance

`llm.llguidance.Constraint` (`src/llm/llguidance.zig`) compiles a grammar
(`json_schema` | `regex` | `lark` | composite `llguidance`) and adapts it to
the seam. llguidance was chosen over the alternatives (§8) because it is
the engine behind vLLM/SGLang-class structured output: full JSON-schema
coverage, a Lark-variant CFG language, regex, per-step masks over the whole
vocab in ~10–50 µs via a token-trie + derivative-based lexer, and a
maintained C FFI.

Mechanics worth pinning:

- **Tokenizer bridge** (`buildVocab`, `src/llm/llguidance.zig:245`): the
  engine needs every token's RAW bytes. Byte-BPE tokens are byte-decoded;
  SPM pieces are unescaped (`▁` → space) and `<0xXX>` byte tokens become
  their byte. Control tokens carry toktrie's `0xFF` special marker
  (`:201`) — BPE recognizes them by the `<|...|>` marker shape (the same
  set `encodeWithSpecials` resolves atomically; byte-BPE has no attribute
  table), SPM by its GGUF-declared `control`/`unknown` attrs. The marker is
  a correctness feature, not bookkeeping: without it, a JSON string value
  containing the text `<|im_end|>` would let the sampler emit the actual
  control token and silently end the turn mid-object. `0xFF` never occurs
  in valid UTF-8, so no ordinary token collides.
- **Vocabulary padding**: models pad `config.vocab_size` past the tokenizer
  vocab (Qwen3: 151 936 vs 151 669). The mask is sized to the MODEL vocab
  (`Options.n_vocab`); padding ids get empty token bytes, which the trie
  never matches — padding logits are permanently `-inf` under a constraint.
- **Canonical re-tokenization**: the engine's `tokenize_fn` callback runs
  Fucina's own tokenizer (`Bridge`, `:214` — BPE `encodePlainAppend`, SPM
  `encodeRaw`), which is what makes grammar-forced byte strings tokenize
  canonically (and `forcedTokens` non-empty, §5). Built without llguidance's
  `rayon` feature, the callback only ever runs on the calling thread.
- **Terminal behavior**: grammar complete → the mask forces
  `Options.eos_token` (chat passes the template's stop-marker id, the
  completion path defaults `--stop` to EOS); a mid-decode matcher failure
  degrades the same way (`vtProcess`, `:458`) so a stream always
  terminates cleanly, while an invalid grammar fails `init` loudly with the
  engine's diagnostic.
- **Build gating**: `-Dllguidance=true` (default off) runs
  `cargo build --release` in `vendor/llguidance` and links the staticlib
  into the qwen3/gemma4 examples and the llm test roots. Off, a stub
  `Constraint` keeps every caller compiling and `init` returns
  `error.LlguidanceNotEnabled`; no Rust symbol is referenced, the build
  stays pure Zig. The staticlib keeps Rust's `panic = unwind` (the FFI's
  `catch_unwind` converts grammar panics to error strings — `abort` would
  lose that), which needs an unwinder at link time: libSystem covers macOS,
  and non-macOS targets link Zig's bundled LLVM libunwind
  (`configureLlguidance` in `build.zig`) — hermetic, no system libgcc_s
  dependency. Vendoring policy (crates-only + pinned `Cargo.lock`,
  manifest deviations, the full-offline `cargo vendor` recipe, the update
  procedure): `vendor/llguidance/README.md`. Provenance:
  `docs/THIRD-PARTY-NOTICES.md`.

## 4. Composition with speculative decoding — no rollback needed

The obvious integration worry: speculative decoding samples *hypothetical*
continuations, and a grammar is stateful — surely the constraint needs a
rollback primitive for rejected drafts (llguidance even provides one,
`llg_matcher_rollback`). It does not, and the reason is a property of
Fucina's verify loop worth stating precisely:

> **Every `Sampler.next` result is a committed token.** The verify loop
> (`speculative/core.zig`, `verifyStep`) samples row *i* only after rows
> `0..i-1`'s tokens are appended to history, and the token sampled at row
> *i* is itself committed immediately — as an accepted draft token, as the
> correction token (first mismatch, which ends the row loop), or as the
> bonus token. No sampled token is ever discarded by the decoder.

Since `commit` fires inside `next`, the matcher state advances in lockstep
with history by construction — there is nothing to roll back. A draft token
the grammar forbids is masked to `-inf` at its verify row, so the sampled
token *cannot* equal it; the `sampled == draft` comparison fails; the
sampled token IS the correction. Rejection sampling semantics are exactly
preserved, and the constrained speculative stream equals the constrained
plain stream token-for-token (given bitwise-equal logits — the same §13.9
caveat as unconstrained speculation).

The turn boundary is the one place a sampled token does *not* enter
history: the stop marker itself, and a token completing a text stop
sequence. Both are dropped **identically by every path** (plain `send`,
`sendSpec`'s `TurnGate` — which truncates drafts at stop tokens so the
boundary token only ever arrives as a sampled correction/bonus — and
`sendBatch`'s `sampleStep`), so processor state after any turn is the same:
post-stop, re-armed by the next turn's `reset`.

Proofs in-tree (`src/llm/chat_tests.zig`): constrained plain == constrained
speculative, greedy and sampled with a persistent RNG, both for a plain
mask processor and for a structural (forced-span) processor; commit-log
equality is asserted, not just stream equality.

## 5. Grammar-driven drafting — `ConstrainedSource`

Without help, a constraint *hurts* speculation: the SAM/recycling cascade
drafts unconstrained text, the mask rejects it, acceptance collapses and
the CostGate rightly mutes the feature (measured: 0% acceptance,
`off 18/21` steps on JSON-schema chat). But the grammar knows things the
cascade cannot, and `speculative/constrained.zig:36` turns that knowledge
into drafts. It wraps any inner `DraftSource` with a structural processor
(the two optional vtable hooks, both **deterministic pure lookaheads** —
verified against the vendored Rust: `Matcher::compute_ff_tokens` and
`validate_tokens` do not mutate parser state):

- **Forced spans draft themselves** (`forcedTokens`, backed by
  llguidance's fast-forward tokens). When the grammar mandates a unique
  continuation — `", "population": ` after a JSON key closes — those tokens
  are the draft. The masked sampler can only select the forced token at
  each row, so the span verifies with **acceptance probability 1**: one
  batched forward commits it all, plus the free bonus token.
- **Certainly-rejected drafts die pre-verify** (`validPrefixLen`, backed by
  token validation). On free-choice steps the inner cascade proposes and
  the draft is truncated at its first grammar-invalid token — tokens that
  would be masked at their verify row can only waste verify compute and
  drag the cascade's per-source acceptance gates down.

Accounting stays honest through a small core extension:
`DraftSource.truncatePending` (`speculative/core.zig:93`), the generic
"your just-returned draft was shortened" notification. The cascade shrinks
its pending acceptance window; the chat `TurnGate` now uses the same vtable
entry instead of reaching into the concrete cascade (it is fully
source-generic); `ConstrainedSource` forwards truncations only when the
live draft came from the inner source — forced drafts carry no pending
accounting anywhere. `wantsTopK` mirrors the inner source, so wrapping
never makes the decoder compute top-k feedback nobody consumes.

Losslessness is untouched: drafts never decide *what* is committed, and
both hooks are deterministic, so the source stays deterministic. The wiring
is automatic — `chat.Conversation` wraps its cascade whenever the installed
processor `hasStructure()` (`src/llm/chat.zig:207`), and the qwen3 `--spec`
path does the same.

Measured (Qwen3-0.6B-Q8_0, greedy JSON-schema chat, M1 Max):

| | drafted | accepted | gate | tok/step |
| --- | --- | --- | --- | --- |
| constraint, no drafting layer | 10 | 0 (0%) | muted after 3 steps | 1.00 |
| constraint + `ConstrainedSource` | 6 | 5 (83%) + bonus | never off | 1.24 |

Output byte-identical in both rows and to the non-speculative constrained
run. The `fallback` steps that remain are forced spans shorter than the
decoder's `min_draft` (single forced tokens take the plain step) — headroom
for a future `min_draft`-aware forced-span policy.

## 6. Multi-stream — `Constraint.clone()`

A constraint is single-stream state, so N-stream decode needs N matcher
states. Re-running `init` per stream would rebuild the vocab trie and
recompile the grammar; `clone()` (`src/llm/llguidance.zig:385`) instead
deep-clones the matcher (initial state if cloned after init/reset),
reference-counts the tokenizer handle, and **borrows the tokenize bridge**
from the original — the original must outlive its clones, which every
current caller satisfies structurally (the base constraint lives in `main`,
clones in the stream loop).

Consumers:

- `chat.sendBatch`: per-conversation processors arrive via each
  conversation's own `Options.logit_processor`; the batch validator rejects
  a shared pointer (`error.SharedBatchProcessor`). Constrained lockstep
  output is proven equal to individual constrained sends
  (`chat_tests.zig`; n below the m-dependent kernel thresholds, so
  bitwise).
- qwen3 `--streams N` + a grammar flag: one clone per stream, reset per
  bench pass, both arms (lockstep and sequential) constrained identically —
  the harness's token-for-token cross-check passes on the real model.

## 7. Operational notes and honest caveats

- **Greedy + an unbounded grammar field loops.** `{"population": <integer>}`
  under `--temp 0`: the model wants to write `2.1 million`, the grammar
  forbids `.`, greedy re-picks `0` forever — the grammar cannot force
  termination inside a field whose continuation is always legal, and argmax
  never chooses to stop. This is constraint semantics, not a masking bug.
  Mitigations, in order: bound fields (`maximum`, `maxLength`, `{m,n}`),
  sample instead of greedy, keep `repeat_penalty` on. Documented in
  `RUNNING-MODELS.md`.
- **Constrain the whole reply** — on reasoning models combine with
  `--no-think`, or the grammar forbids the `<think>` preamble the model
  wants to emit.
- **SPM `tokenize_fn` nuance**: SPM re-tokenization goes through
  `encodeRaw`, which applies `add_space_prefix` when the model sets it.
  Gemma sets it false, so the shipped families are unaffected; a
  space-prefixing SPM model would degrade only forced-token *drafting*
  granularity (mask correctness is trie-based and unaffected).
- **gemma4 is wired but not e2e-validated** (no gemma GGUF on the dev disk
  at the time of writing); its SPM bridge is covered by the gated unit
  tests (attrs marking, byte-fallback tokens, control-token exclusion).
- llguidance's per-grammar `temperature` extension is deliberately ignored
  (the Matcher API drops it; sampling knobs stay the user's).
- The engine adds ~15 MB to constrained binaries (Rust staticlib, stripped)
  and one opt-in toolchain requirement (cargo ≥ 1.87). Mask computation is
  off the model's critical path in practice (~tens of µs vs ms-scale
  forwards).

## 8. Adjudications

- **Per-loop mask wiring (llama.cpp's shape: grammar applied inside each
  sampler chain construction)** — rejected. Fucina has five-plus decode
  loops (plain chat, spec plain/verify, batch, runner completion/bench);
  the sampler-hosted hook constrains all of them with one seam and keeps
  the invariant "every path samples identically" a structural fact rather
  than a per-loop obligation.
- **Rollback-based speculative integration** (`llg_matcher_rollback` on
  draft rejection) — rejected as unnecessary: Fucina's verify loop commits
  every sampled token (§4), so there is never un-committed matcher state.
  Would reopen only if the decoder ever moved to sampling rows beyond an
  uncommitted prefix (e.g. tree/beam speculation).
- **Porting a grammar engine (GBNF-style) into Zig** — rejected for
  scope: full JSON-schema semantics + a maintained Earley/lexer stack is a
  project-sized dependency to re-own, and constrained decoding is pure
  CPU-side logic where the FFI boundary costs nothing per token. The
  hand-written extern layer is ~15 declarations against a checked-in
  header, with an ABI round-trip in the gated tests. Would reopen if the
  Rust toolchain requirement became a real adoption barrier even as
  opt-in.
- **Fast-forward tokens as forced *injection*** (commit ff tokens
  directly, skipping sampling) — rejected: it changes the
  one-RNG-draw-per-committed-token accounting that the lossless
  speculation contract and the chat equivalence proofs rest on. Drafting
  them instead (§5) captures the same forward-pass savings *within* the
  existing contract, at the cost of one extra logits row per span.
- **Masking after top-k truncation** (filter the 256-candidate list
  instead of the full row) — rejected: cheaper per step but wrong — when
  all top-256 candidates are grammar-invalid the correct next token lies
  outside the truncated set, and the greedy path bypasses top-k entirely.
  The full-row mask is O(vocab/32) words of bit-tests; not worth a
  correctness cliff.

## 9. Test and validation map

| Claim | Where proven |
| --- | --- |
| Seam semantics (mask on both paths, one commit per selection, all-masked failure, penalty composition) | `src/llm/logit_processor_tests.zig` |
| Grammar walks, stop forcing, reset, special-token exclusion, JSON-schema over byte vocab, SPM attrs + byte-fallback, invalid grammars, ABI round trip | `src/llm/llguidance_tests.zig` (gated: skips unless `-Dllguidance=true`) |
| Structural hooks are pure lookaheads; clone independence | `src/llm/llguidance_tests.zig` |
| Combinator policy (forced preemption, invalid-prefix truncation, pending accounting, top-k mirroring) | `src/llm/speculative/constrained_tests.zig` |
| Constrained plain == speculative (greedy + sampled), forced spans drafted AND accepted, per-turn reset, batch == sequential per-stream constraints, shared-processor guard | `src/llm/chat_tests.zig` |
| Doc snippets (incl. the flag-gated llguidance snippet) | `zig build snippet-check` (§2.7 convention) |
| E2E schema/regex conformance, `--spec` byte-parity + acceptance, `--streams` cross-check | qwen3 runner on Qwen3-0.6B-Q8_0 (2026-07-11; grammar commands in `RUNNING-MODELS.md` §"Constrained decoding", the `--spec`/`--streams` commands in `examples/qwen3/README.md`) |
| Linux staticlib link + full gated suite (x86-64 glibc) | CI llguidance leg (`ci.yml`, ubuntu; §2.8) — first proven natively on the dev rig, 2026-07-11 |

Reference documentation: REFERENCE.md §13.6 (seam + engine), §13.9.6
(drafting), §2.2 (`-Dllguidance`), §13.8 (chat wiring).
