# Video 13 — Inference tricks (3:00)

*Series: Forging Deep Learning in Zig · Source: ../13-inference-tricks.md*

## Logline

Three tricks that make the same transformer faster, bigger-than-RAM, and
schema-guaranteed — without changing what it says: draft-model-free
speculative decoding behind a never-a-loss gate, MoE expert streaming that
decodes a 142 GB model on a 64 GB machine, and constrained decoding that
composes with speculation. Each one is a seam, not a hack — and `lmserve`
puts them all behind one OpenAI-compatible command.

## Takeaways

1. The tricks are seams: `DraftSource`, `LogitProcessor`, the tiered expert
   store — small interfaces placed where every decode path already funnels.
   That placement is *why* they compose (a grammar can become a drafter).
2. "Never-a-loss" is a gate property, not a drafting property: deterministic
   drafts make speculation lossless (drafts change how fast tokens commit,
   never which tokens), and a measured cost gate gives a 0.98–0.99×
   floor with 1.1–2.3× task-dependent wins — 2.3× being the
   verbatim-repetition microcase.
3. A 142 GB mixture model decodes **bit-identically** on a 64 GB machine by
   paging routed experts — pinned hot set → per-layer LRU → `pread` — and a
   persisted usage histogram auto-pins the hot set, so the engine gets
   faster the more it is used.

## Script

### [0:00–0:22] Three promises, one theme

**VO:** Three promises about one language model. It decodes faster —
provably committing the exact tokens plain decoding would. A 142-gigabyte
model runs on a 64-gigabyte machine. And its reply is guaranteed to match
your JSON schema — at sampling time, not by retry. None of these are hacks.
Each is a seam. And seams compose.

**Visual:** Three-panel title card, panels appearing in sync with the VO:
"FASTER — lossless" / "142 GB on 64 GB" / "JSON, guaranteed". As the VO says
"seams compose", plug-socket connectors draw between the panels. Small
persistent caption: "same model · same tokens · chapter 13".

**Overlay:** "faster · cheaper · controllable — without changing what it
says (ch. 13 intro)".

### [0:22–1:10] Trick 1 — speculation with no draft model

**VO:** First: speculative decoding — without a draft model. Decode reads
every weight to produce one token. So guess a few tokens from text the model
has already seen: a suffix automaton over the conversation finds the longest
repeated suffix and proposes what followed it. One batched forward checks
all the guesses. Because the drafter is deterministic, acceptance is plain
token equality — and the first mismatch is itself the correction. Drafts
change how fast tokens commit, never which tokens. A measured cost gate
shuts the trick off when it can't pay. Honest numbers: tasks it can't
accelerate run at 0.98 to 0.99x. Tasks it can, 1.1 to 2.3x — and 2.3 is the
verbatim-repetition microcase, not the typical day.

**Visual:** Diagram first: token stream `5 6 7 5 6`; the suffix `[5,6]`
highlights at the end AND at its earlier occurrence; an arrow pulls the
tokens that followed — `7 5 6` — forward as the draft (per §13.4 /
`docs/REFERENCE.md` §13.9.2). Then code shot:
`src/llm/speculative/core.zig:9–20` — the normative losslessness header —
with "sampled == draft[i]" and "the sampled token IS the correction token"
highlighted in sequence. Close on a numbers card quoting §13.5's table:
"grounded copy 1.47× (pre-tokenizer-fix encoding; same prompt 1.04×
post-fix) · code edit 1.12× · verbatim repetition 2.3× · free-form 0.99×
(gate auto-off)".

**Overlay:** On the numbers card, persistent caption: "M1 Max · ReleaseFast ·
Qwen3-0.6B-Q4_K_S · greedy · docs/SPECULATIVE.md §10" and "floor 0.98–0.99× ·
wins 1.1–2.3× · **2.3× = verbatim-repetition microcase, 100% acceptance**" ·
"'never-a-loss' is a gate property, not a drafting property".

### [1:10–1:50] Trick 2 — 142 GB on a 64 GB machine (showcase)

**VO:** Second, the showcase. Qwen3-235B at Q4_K_M is 142 gigabytes of
weights. This M1 Max has 64. But a mixture-of-experts model routes only a
few experts per token, so Fucina keeps the dense weights resident and pages
experts from disk: a pinned hot set, a per-layer LRU, then pread. Output
stays bit-identical to the resident path. Cold, 0.66 tokens per second at a
52-percent hit rate. On the second run a usage histogram auto-pins the 944
hottest experts — 0.81. The engine gets faster the more it is used. Not
interactive. But it runs — on your machine.

**Visual:** Terminal type-on (do not execute; see production notes) of the
§13.8 command:
```
zig build qwen3 -Doptimize=ReleaseFast -- \
  models/Qwen3-235B-A22B-Instruct-2507-Q4_K_M-00001-of-00003.gguf \
  --prompt "The three most important ideas in computer science are" \
  --gen 64 --moe-stream --moe-cache-mb=20480
```
Then a three-tier diagram: "pinned hot set → per-layer LRU → `pread`", with
a router icon feeding it and "the OS sees pages, not experts" as the
footnote (§13.8 ML note). Then code shot:
`src/exec/expert_store.zig:872–890` — the acquire resolution loop — with
the pinned-hit, LRU-hit, and miss branches highlighted in sequence.

**Overlay:** "142 GB GGUF · 64 GB RAM · ~24 GB peak RSS @ 20 GB expert
budget" · "cold 0.66 tok/s @ 52% hits → warm 0.81 tok/s @ 59% (auto-pin:
944 experts, 10.7 GB) · M1 Max · docs/RUNNING-MODELS.md" · "bit-identical to
the resident path · sub-1 tok/s = the price of running it at all".

### [1:50–2:25] Trick 3 — the grammar becomes a drafter

**VO:** Third: constrained decoding. Before each sample, the grammar writes
negative infinity over every forbidden token; softmax renormalizes over
what's left. The hook lives inside the sampler, so every decode path gets
it — plain chat, batch, speculation. And here the seams pay off. A grammar
naively kills speculation: unconstrained drafts get masked, acceptance hits
zero. So the grammar becomes a drafter. Spans it forces — the next JSON key,
the punctuation — draft themselves and verify with probability one.
Measured: acceptance zero to 83 percent, output byte-identical.

**Visual:** Code shot: `src/llm/logit_processor.zig:35–64` — the
`LogitProcessor` vtable — highlighting `process` ("mask before the
pipeline"), `commit` ("observe after"), then the two optional structural
hooks `forcedTokens` / `validPrefixLen`. Then code shot:
`src/llm/speculative/constrained.zig:36–56` — the entire public surface of
`ConstrainedSource`, two functions. Close on the §13.6 command as a terminal
shot (runnable if the Q8_0 model is present):
```
zig build qwen3 -Dllguidance=true -Doptimize=ReleaseFast -- models/Qwen3-0.6B-Q8_0.gguf \
  --chat "Give me facts about Paris." --no-think \
  --json-schema '{"type":"object","properties":{"city":{"type":"string"},"population":{"type":"integer","maximum":99999999}},"required":["city","population"],"additionalProperties":false}'
```

**Overlay:** "one hook in the sampler → every decode path constrained
(docs/CONSTRAINED-DECODING.md)" · numbers card caption: "without
ConstrainedSource: 10 drafted / 0 accepted, gate muted, 1.00 tok/step ·
with: 83% acceptance, 1.24 tok/step · output byte-identical ·
Qwen3-0.6B-Q8_0, greedy JSON-schema chat, M1 Max ·
docs/CONSTRAINED-DECODING.md §5".

### [2:25–2:42] One command to serve it

**VO:** All of it meets the network in lmserve: an OpenAI-compatible server
over the same engine. One command, and any OpenAI client talks to your
model — schema constraints, KV reuse, the works — with parameters it can't
honor rejected loudly, never dropped.

**Visual:** Terminal shot (runnable) of the §13.10 command:
```
zig build lmserve -Dllguidance=true -Doptimize=ReleaseFast -- \
  models/Qwen3-0.6B-Q8_0.gguf --port 8080
# then point any OpenAI client at http://localhost:8080/v1
```
Hold on the running server; a side card lists the flags: `--json-schema via
response_format · --kv-slots · --kv-cache-dir`.

**Overlay:** "verified end-to-end against openai-python 2.45.0 (qwen3,
2026-07-12) · docs/LMSERVER.md" · "unmappable params → 400/501 naming the
param — never silently dropped".

### [2:42–3:00] The through-line, and what's next

**VO:** The through-line: a rewindable cache, deterministic drafts, measured
gates. Every trick is either provably lossless or gated so it can never
lose — and when it can't win, a gate says so with numbers. Next: fewer bytes
per weight. The low-bit frontier.

**Visual:** Single closing card, four words appearing in a chain:
"truncate → rewind → verify → gate" (the chapter's own through-line, §13.10
closing paragraph). End card: series title, "Next: 14 — The low-bit
frontier", chapter link `docs/course/13-inference-tricks.md`.

**Overlay:** "provably lossless — or gated so it can never lose" · end card:
"Next: The low-bit frontier".

## Asset list

**Code shots (repo files, exact ranges — ranges verified 2026-07-12,
re-verify at record time):**
- `src/llm/speculative/core.zig:9–20` — the normative losslessness header
  (doc comment).
- `src/exec/expert_store.zig:872–890` — the acquire resolution loop
  (pinned hit / LRU hit / miss collect).
- `src/llm/logit_processor.zig:35–64` — the `LogitProcessor` vtable with
  optional structural hooks.
- `src/llm/speculative/constrained.zig:36–56` — `ConstrainedSource`'s
  two-function public surface (`init` at :47, `source` at :54–56).

**Terminal shots:**
- Type-on ONLY (do not execute): the 235B `--moe-stream` command exactly as
  in §13.8. All its numbers are quoted from `docs/RUNNING-MODELS.md`, not
  re-measured.
- Runnable: the qwen3 `--json-schema` command (§13.6) and the `lmserve`
  command (§13.10), both on Qwen3-0.6B-Q8_0. If live output is recorded, do
  NOT substitute live throughput numbers for the documented ones — the
  documented numbers stay as rendered cards with their captions.

**Diagrams to render (one sentence each):**
- SAM draft diagram: stream `5 6 7 5 6`, suffix `[5,6]` matched at its
  earlier occurrence, draft `{7,5,6}` pulled forward (§13.4).
- Expert-streaming tiers: router → "pinned hot set → per-layer LRU →
  `pread`", footnote "the OS sees pages, not experts" (§13.8 ML note).
- Three-promise title triptych with plug-socket "seams compose" connectors.
- Closing chain card: "truncate → rewind → verify → gate".
- Speculation numbers card and constrained-decoding numbers card, all
  figures quoted from `docs/SPECULATIVE.md` §10 and
  `docs/CONSTRAINED-DECODING.md` §5 as reproduced in chapter §13.5/§13.6.
- End card with next-episode teaser.

**External downloads (weights are NOT in the repo):**
- Qwen3-0.6B GGUF, Q8_0 (for the runnable constrained + lmserve shots) and
  optionally Q4_K_S (only if a live `--spec` B-roll shot is wanted) — GGUF
  releases per `docs/RUNNING-MODELS.md`.
- Qwen3-235B-A22B-Instruct-2507 Q4_K_M GGUF (142 GB, 3 split parts) — NOT
  required: the segment is a type-on plus quoted-numbers card; download only
  if a genuine live run is affordable.

## Production notes

- **Tone:** matter-of-fact wonder. The 142-GB-on-64-GB beat is the episode's
  showcase — let the terminal command and the RSS number carry it. "Not
  interactive. But it runs" is honest framing, not an apology; deliver it
  flat, like the chapter does ("the price of running the 235B on your own
  machine at all").
- **Caveats are load-bearing and MUST NOT be cut:** (a) whenever 2.3× is on
  screen or spoken, "verbatim-repetition microcase, 100% acceptance" and the
  0.98–0.99× floor must be visible, and the 1.47× row keeps its
  "pre-tokenizer-fix encoding / 1.04× post-fix, same prompt" qualifier
  (§13.5: single-prompt speedups are fragile); (b) the speculation numbers card keeps
  its full caption (M1 Max · ReleaseFast · Qwen3-0.6B-Q4_K_S · greedy ·
  docs/SPECULATIVE.md §10); (c) the MoE numbers keep "M1 Max · 64 GB ·
  ~24 GB peak RSS @ 20 GB expert budget · docs/RUNNING-MODELS.md" and the
  sub-1-tok/s honesty line; (d) the 0%→83% card keeps "Qwen3-0.6B-Q8_0 ·
  greedy JSON-schema chat · M1 Max"; (e) "byte-identical" and
  "bit-identical" are exact claims from the docs — do not soften or inflate
  them.
- **Do not execute the 235B command on camera** unless the 142 GB download
  and a real M1-Max-class 64 GB machine are actually available; the type-on
  + quoted card is the default plan. Never present a live run's tok/s as the
  documented 0.66/0.81 figures.
- **If the cut runs long, trim in this order:** the SAM diagram's animation
  (land it as one frame), then the expert-tier diagram's dwell, then the
  `ConstrainedSource` code shot (the VO carries it). Never trim a caveat
  overlay, the losslessness header shot, or the lmserve command.
- **Numbers appearing in the video and their sources (nothing else may be
  quantified):** 0.98–0.99× floor and 1.1–2.3× range, 1.47×/1.12×/2.3×/0.99×
  task rows (§13.5, `docs/SPECULATIVE.md` §10); 142 GB / 64 GB / ~24 GB RSS /
  20 GB budget / 0.66→0.81 tok/s / 52%→59% hit rate / 944 experts / 10.7 GB
  (§13.8, `docs/RUNNING-MODELS.md`); 0%→83% acceptance, 10/0 vs 6/5 drafted/
  accepted, 1.00→1.24 tok/step (§13.6, `docs/CONSTRAINED-DECODING.md` §5);
  openai-python 2.45.0 verification (§13.10, `docs/LMSERVER.md`).
- The composition claim in segment 3 ("every decode path gets it") is the
  chapter's sampler-hosted-hook argument (§13.6); keep the phrase "inside
  the sampler" intact — it is the design point.
- The next-episode teaser ("The low-bit frontier") matches Video 14's title
  and must survive edits.
